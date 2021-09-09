module ScalableHrlEs

using HrlMuJoCoEnvs

import ScalableES
using ScalableES: NoiseTable, forward, LyceumMuJoCo, Policy

using MPI: MPI, Comm
using Flux
import StatsBase
import Statistics: mean, std

include("Util.jl")
include("Policy.jl")
include("Optim.jl")
include("Obstat.jl")

function run_hrles(name::String, cnn, pnn, envs, comm::Comm; 
                   gens=150, npolicies=256, episodes=3, σ=0.02f0, nt_size=250000000, η=0.01f0)
    @assert npolicies / size(comm) % 2 == 0 "Num policies / num nodes must be even (eps:$npolicies, nprocs:$(size(comm)))"

    steps = 500
    intervals = 50

    println("Running ScalableEs")
    tblg = ScalableES.TBLogger("tensorboard_logs/$(name)", min_level=ScalableES.Logging.Info)

    env = first(envs)
    obssize = length(ScalableES.obsspace(env))

    println("Creating policy")
    p = HrlPolicy(cnn, pnn)
    ScalableES.bcast_policy!(p, comm)

    println("Creating noise table")
    nt, win = NoiseTable(nt_size, length(p.cπ.θ), σ, comm)

    obstat = HrlObstat(obssize, obssize, 1f-2)
    opt = ScalableES.isroot(comm) ? HrlAdam(length(p.cπ.θ), length(p.pπ.θ), η) : nothing
    f = (nns, e, obmean, obstd) -> hrl_eval_net(nns, e, obmean, obstd, intervals, steps, episodes)

    println("Initialization done")
    ScalableES.run_gens(gens, name, p, nt, f, envs, npolicies, opt, obstat, tblg, comm)

    model = ScalableES.to_nn(p)
    ScalableES.@save joinpath(@__DIR__, "..", "saved", name, "model-obstat-opt-final.bson") model obstat opt

    MPI.free(win)
end

function ScalableES.noiseify(π::Policy, nt::NoiseTable, ind::Int)
    noise = ScalableES.sample(nt, ind, length(π.θ))
	Policy(π.θ .+ noise, π._re), Policy(π.θ .- noise, π._re), ind
end

function ScalableES.noiseify(π::HrlPolicy, nt::ScalableES.NoiseTable)
    ind = rand(nt, max(length(π.cπ.θ), length(π.pπ.θ)))
    cpos_pol, cneg_pol, _ = ScalableES.noiseify(π.cπ, nt, ind)
    ppos_pol, pneg_pol, _ = ScalableES.noiseify(π.pπ, nt, ind)

    HrlPolicy(cpos_pol, ppos_pol), HrlPolicy(cneg_pol, pneg_pol), ind
end

function ScalableES.optimize!(π::HrlPolicy, ranked::Vector{HrlEsResult{T}}, nt::NoiseTable, optim::HrlAdam, l2coeff::Float32) where T
    ScalableES.optimize!(π.cπ, map(r->r.cres, ranked), nt, optim.copt, l2coeff), ScalableES.optimize!(π.pπ, map(r->r.pres, ranked), nt, optim.popt, l2coeff)
end

ScalableES.make_result_vec(n::Int, ::HrlPolicy) = Vector{HrlEsResult{Float64}}(undef, n)
# don't like the hardcoded +2 but not sure how to get round this
ScalableES.make_obstat(shape, ::HrlPolicy) = HrlObstat(shape, shape, 0f0)


function ScalableES.bcast_policy!(π::HrlPolicy, comm::Comm)
	MPI.Barrier(comm)
    π.cπ = ScalableES.bcast(π.cπ, comm)
	π.pπ = ScalableES.bcast(π.pπ, comm)
end

function encode_prim_obs(obs::Vector{T}, env, targ_vec, targ_dist) where T<:AbstractFloat
    cp = copy(obs)
    cp[1:2] = angle_encode_target(targ_vec, LyceumMuJoCo._torso_ang(env))
    cp[3] = targ_dist / 1000
    cp
end

function hrl_eval_net(nns::Tuple{Chain, Chain}, env, (cobmean, pobmean), (cobstd, pobstd), cintervals::Int, steps::Int, episodes::Int)
    cnn, pnn = nns
    
    cobs = Vector{Vector{Float64}}()
    pobs = Vector{Vector{Float64}}()

	cr = 0
    pr = 0
	step = 0

    rel_target = zeros(2)
    abs_target = zeros(2)
    pos = zeros(2)
    d_old = 0f0

    rewarded_prox = false  # rewarded primitive for being close to target

	for i in 1:episodes
		LyceumMuJoCo.reset!(env)
        died = false
		for i in 1:steps
            ob = LyceumMuJoCo.getobs(env)

            if i % cintervals == 0  # step the controller
                rel_target = forward(cnn, ob, cobmean, cobstd) * 5                  
                abs_target = rel_target + pos
                rewarded_prox = false
                push!(cobs, ob)
            end

            rel_target = abs_target - pos  # update rel_target each time
            # pob = vcat(rel_target, ob)
            pob = encode_prim_obs(ob, env, rel_target, d_old)

			act = forward(pnn, pob, pobmean, pobstd)
			LyceumMuJoCo.setaction!(env, act)
			LyceumMuJoCo.step!(env)

			step += 1
			push!(pobs, pob)  # propogate ob recording to here, don't have to alloc mem if not using obs
			
            # calculating rewards
            pos = HrlMuJoCoEnvs._torso_xy(env)
            d_new = HrlMuJoCoEnvs.sqeuclidean(pos, abs_target)

            pr += (d_new - d_old) / LyceumMuJoCo.timestep(env)
            if d_new < 2^2 && !rewarded_prox
                pr += 5000
                rewarded_prox = true
            end
            
            d_old = d_new
            
			if LyceumMuJoCo.isdone(env) 
                died = true
                break
             end
		end
        cr += LyceumMuJoCo.getreward(env)  # only care about the last reward for controller (should check if robot died)
	end
	# @show rew step
	(cr / episodes, pr / episodes), step, (cobs, pobs)
end

end # module
