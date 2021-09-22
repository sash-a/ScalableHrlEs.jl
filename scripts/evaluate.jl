include("../src/ScalableHrlEs.jl")
using .ScalableHrlEs
using ScalableES

using MuJoCo
using LyceumMuJoCo
using HrlMuJoCoEnvs

using Flux

using BSON: @load
using StaticArrays
using ArgParse

function evalsaved(runnames, gen, intervals::Int, cdist::Float32)
    mj_activate("/home/sasha/.mujoco/mjkey.txt")
    max_rews = []
    for runname in runnames
        @load "saved/$(runname)/model-obstat-opt-gen$gen.bson" model obstat opt
        env = HrlMuJoCoEnvs.AntGatherEnv()

        # states = collectstates(model, env, obmean, obstd)
        obmean = ScalableHrlEs.mean(obstat)
        obstd = ScalableHrlEs.std(obstat)
        cnn, pnn = model

        # getting world age errors so manually reconstructing pnn
        p, _ = Flux.destructure(pnn)
        pnn = Chain(Dense(46 + 3, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                    Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                    Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                    Dense(256, 8, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                    x -> x .* 30)
        _, re = Flux.destructure(pnn)

        pnn = re(p)

        maxr, (cr, pr) =  evaluate((cnn, pnn), env, obmean, obstd, intervals, 500, 10, cdist; cforward=ScalableHrlEs.onehot_forward)
        push!(max_rews, maxr)
        @show maxr cr pr
    end

    print("average early stopped rew = $(sum(max_rews)/length(max_rews))")
    # env = HrlMuJoCoEnvs.AntGatherEnv(viz=true)
end
   
function evaluate(nns::Tuple{Chain, Chain}, env, (cobmean, pobmean), (cobstd, pobstd), 
                  cintervals::Int, steps::Int, episodes::Int, cdist::Float32; cforward=ScalableHrlEs.onehot_forward)
    cnn, pnn = nns
    
    cobs = Vector{Vector{Float64}}()
    pobs = Vector{Vector{Float64}}()

	cr = 0
    pr = 0
	step = 0

    rewarded_prox = false  # rewarded primitive for being close to target
    sqrthalf = sqrt(1/2)

    sensor_span = hasproperty(env, :sensor_span) ? env.sensor_span : 2 * π
    nbins = hasproperty(env, :nbins) ? env.nbins : 8

    max_rew = 0

	for ep in 1:episodes
		LyceumMuJoCo.reset!(env)
        died = false
        
        pos = zeros(2)
        rel_target = zeros(2)
        abs_target = zeros(2)
        
        targ_start_dist = 0  # dist from target when controller suggests position
        d_old = 0f0

		for i in 0:steps - 1
            ob = LyceumMuJoCo.getobs(env)

            if i % cintervals == 0  # step the controller
                c_raw_out = cforward(cnn, ob, cobmean, cobstd, cdist, LyceumMuJoCo._torso_ang(env), sensor_span, nbins)
                rel_target = ScalableHrlEs.outer_clamp.(c_raw_out, -sqrthalf, sqrthalf)
                # rel_target = outer_clamp.(rand(Uniform(-cdist, cdist), 2), -sqrthalf, sqrthalf)
                abs_target = rel_target + pos
                rewarded_prox = false
                push!(cobs, ob)

                targ_start_dist = d_old = HrlMuJoCoEnvs.euclidean(abs_target, pos)
            end

            rel_target = abs_target - pos  # update rel_target each time
            # pob = vcat(rel_target, ob)
            pob = ScalableHrlEs.encode_prim_obs(ob, env, rel_target, d_old / 1000)
			act = ScalableES.forward(pnn, pob, pobmean, pobstd)
			LyceumMuJoCo.setaction!(env, act)
			LyceumMuJoCo.step!(env)

			step += 1
			push!(pobs, pob)  # propogate ob recording to here, don't have to alloc mem if not using obs
			
            # calculating rewards
            pos = HrlMuJoCoEnvs._torso_xy(env)
            d_new = HrlMuJoCoEnvs.euclidean(pos, abs_target)

            cr += LyceumMuJoCo.getreward(env)
            # pr += (d_old - d_new) / LyceumMuJoCo.timestep(env)
            # if d_new < 1^2 && !rewarded_prox
            #     rewarded_prox = true
            # end
            pr += 1 - (d_new / targ_start_dist)
            pr += d_new < 1 ? 1 : 0
            
            d_old = d_new
            
			if LyceumMuJoCo.isdone(env) 
                died = true
                break
             end

             max_rew = max(cr, max_rew)
		end
	end
	# @show rew step
	max_rew/episodes ,(cr / episodes, pr / episodes)
end

evalsaved(["remote/gatherofficial/antgather-official-50_3"], 600, 25, 4f0)

file = "saved/remote/gatherofficial/antgather-official-50_0/model-obstat-opt-gen500.bson"
BSON.load()
@load "saved/remote/gatherofficial/antgather-official-50_0/model-obstat-opt-gen500.bson" model obstat opt
cnn, pnn = model
layers = pnn.layers
typeof(layers)
@show cnn(zeros(46))
@show pnn
@show pnn(zeros(49))