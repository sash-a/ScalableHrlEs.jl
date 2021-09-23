include("../src/ScalableHrlEs.jl")
using .ScalableHrlEs
using ScalableES

using MuJoCo
using LyceumMuJoCo
using LyceumMuJoCoViz
using HrlMuJoCoEnvs

using Flux

using BSON:@load
using StaticArrays
using ArgParse
using SharedArrays
using Distributed

function runsaved(runname, gen, intervals::Int, cdist::Float32)
    mj_activate("/home/sasha/.mujoco/mjkey.txt")

    @load "saved/$(runname)/model-obstat-opt-gen$gen.bson" model obstat opt
    obmean = ScalableHrlEs.mean(obstat)
    obstd = ScalableHrlEs.std(obstat)
    cnn, pnn = model
    
    # model = Base.invokelatest(ScalableES.to_nn, p)
    
    # env = HrlMuJoCoEnvs.AntGatherEnv(viz=true)
    env = HrlMuJoCoEnvs.AntGatherEnv()
    latestsforward = (nn, ob, obm, obs, cd, ang, ss, nb) -> Base.invokelatest(ScalableHrlEs.onehot_forward, nn, ob, obm, obs, cd, ang, ss, nb)
    Base.invokelatest(ScalableHrlEs.forward, cnn, zeros(length(obsspace(env))), first(obmean), first(obstd), 4, 0, 2 * π, 8)
    Base.invokelatest(ScalableHrlEs.onehot_forward, cnn, zeros(length(obsspace(env))), first(obmean), first(obstd), 4, 0, 2 * π, 8)
    Base.invokelatest(ScalableHrlEs.forward, pnn, zeros(length(obsspace(env)) + 3), last(obmean), last(obstd), 4, 0, 2 * π, 8)

    # states = collectstates(model, env, obmean, obstd)

    @show first(ScalableHrlEs.hrl_eval_net((cnn, pnn), env, obmean, obstd, intervals, 500, 100, cdist, cforward=latestsforward))
    # visualize(env, controller=e -> act(model, e, intervals, obmean, obstd, 1000, cdist))
end

function earlystop_evalnet(nns::Tuple{Chain,Chain}, env, (cobmean, pobmean), (cobstd, pobstd), 
    cintervals::Int, steps::Int, episodes::Int, cdist::Float32; cforward=ScalableHrlEs.onehot_forward)
    cnn, pnn = nns

    cobs = Vector{Vector{Float64}}()
    pobs = Vector{Vector{Float64}}()

    cr = 0
    pr = 0
    step = 0

    rewarded_prox = false  # rewarded primitive for being close to target
    sqrthalf = sqrt(1 / 2)

    max_rs = 0

    sensor_span = hasproperty(env, :sensor_span) ? env.sensor_span : 2 * π
    nbins = hasproperty(env, :nbins) ? env.nbins : 8

    for ep in 1:episodes
        LyceumMuJoCo.reset!(env)
        died = false

        pos = zeros(2)
        rel_target = zeros(2)
        abs_target = zeros(2)

        targ_start_dist = 0  # dist from target when controller suggests position
        d_old = 0f0

        max_r = 0
        ep_cr = 0

        for i in 0:steps - 1
            ob = LyceumMuJoCo.getobs(env)

            if i % cintervals == 0  # step the controller
                c_raw_out = cforward(cnn, ob, cobmean, cobstd, cdist, LyceumMuJoCo._torso_ang(env), sensor_span, nbins; rng=nothing)
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
            act = ScalableES.forward(pnn, pob, pobmean, pobstd; rng=nothing)
            LyceumMuJoCo.setaction!(env, act)
            LyceumMuJoCo.step!(env)

            step += 1
            push!(pobs, pob)  # propogate ob recording to here, don't have to alloc mem if not using obs

            # calculating rewards
            pos = HrlMuJoCoEnvs._torso_xy(env)
            d_new = HrlMuJoCoEnvs.euclidean(pos, abs_target)

            r = LyceumMuJoCo.getreward(env)
            cr += r
            ep_cr += r
            max_r = max(ep_cr, max_r)
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
        end
        max_rs += max_r
    end
    # @show rew step
    max_rs / episodes, (cr / episodes, pr / episodes)
end

s = ArgParseSettings()
    @add_arg_table s begin
    "runname"
        required = true
        help = "run name in saved folder"
    "generation"
        required = true
        help = "generation number to view"
    "--intervals", "-i"
        help = "how often the controller suggests a target"
        arg_type = Int
        default = 200
    "--dist", "-d"
        help = "max distance from agent controller can recommend"
        arg_type = Float32
        default = 4f0
end
args = parse_args(s)

runsaved(args["runname"], args["generation"], args["intervals"], args["dist"])