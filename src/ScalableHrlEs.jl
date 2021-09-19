module ScalableHrlEs
# Author Sasha Abramowitz
# prefixing variables with 'p' or 'c' refers to the primitive or controller version of that variable

using HrlMuJoCoEnvs

import ScalableES
using ScalableES: NoiseTable, forward, LyceumMuJoCo, Policy

using MPI: MPI, Comm
using Flux
import StatsBase
import Statistics: mean, std
using Distributions
using Random
using SharedArrays

using StaticArrays
using BSON
using YAML
using Configurations

include("Util.jl")
include("Config.jl")
include("Policy.jl")
include("Optim.jl")
include("Obstat.jl")

function run_hrles(name::String, cnn, pnn, envs, comm::Union{Comm, ScalableES.ThreadComm}; 
                   gens=150, npolicies=256, interval=200, steps=1000, episodes=3, cdist=4, σ=0.02f0, nt_size=250000000, η=0.01f0, pretrained_path="")
    @assert npolicies / size(comm) % 2 == 0 "Num policies / num nodes must be even (eps:$npolicies, nprocs:$(size(comm)))"

    println("Running ScalableEs")
    tblg = ScalableES.TBLogger("tensorboard_logs/$(name)", min_level=ScalableES.Logging.Info)

    env = first(envs)
    obssize = length(ScalableES.obsspace(env))

    println("Creating policy")
    p = HrlPolicy(cnn, pnn, comm)
    ScalableES.bcast_policy!(p, comm)

    println("Creating noise table")
    nt, win = NoiseTable(nt_size, length(p.cπ.θ), σ, comm)

    obstat = HrlObstat(obssize, obssize + 3, 1f-2)
    
    if !isempty(pretrained_path)
        loaded = BSON.load(pretrained_path, @__MODULE__)
        model = loaded[:model]
        obstat = loaded[:obstat]
        p = HrlPolicy{Float32}(model...)
    end

    opt = ScalableES.isroot(comm) ? HrlAdam(length(p.cπ.θ), length(p.pπ.θ), η) : nothing
    f = (nns, e, obmean, obstd) -> hrl_eval_net(nns, e, obmean, obstd, interval, steps, episodes, cdist)

    println("Initialization done")
    ScalableES.run_gens(gens, name, p, nt, f, envs, npolicies, opt, obstat, tblg, comm)

    model = ScalableES.to_nn(p)
    ScalableES.@save joinpath(@__DIR__, "..", "saved", name, "model-obstat-opt-final.bson") model obstat opt

    if win !== nothing
        MPI.free(win)
    end
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

function ScalableES.optimize!(π::HrlPolicy, ranked::Vector{HrlEsResult{T}}, nt::NoiseTable, optim::HrlAdam, l2coeff::Float32) where T <: AbstractFloat
    ScalableES.optimize!(π.cπ, map(r->r.cres, ranked), nt, optim.copt, l2coeff), ScalableES.optimize!(π.pπ, map(r->r.pres, ranked), nt, optim.popt, l2coeff)
end

ScalableES.make_result_vec(n::Int, ::HrlPolicy, ::Comm) = Vector{HrlEsResult{Float64}}(undef, n)
ScalableES.make_result_vec(n::Int, ::HrlPolicy, ::ScalableES.ThreadComm) = SharedVector{HrlEsResult{Float64}}(n)
# don't like the hardcoded +2 but not sure how to get round this
ScalableES.make_obstat(shape, ::HrlPolicy) = HrlObstat(shape, shape + 3, 0f0)


function ScalableES.bcast_policy!(π::HrlPolicy, comm::Comm)
	MPI.Barrier(comm)
    π.cπ = ScalableES.bcast(π.cπ, comm)
	π.pπ = ScalableES.bcast(π.pπ, comm)
end
function ScalableES.bcast_policy!(::HrlPolicy, ::ScalableES.ThreadComm) end

function encode_prim_obs(obs::Vector{T}, env, targ_vec, targ_dist) where T<:AbstractFloat
    vcat(angle_encode_target(targ_vec, LyceumMuJoCo._torso_ang(env)), [targ_dist], obs)
    # cp = copy(obs)
    # cp[1:2] = angle_encode_target(targ_vec, LyceumMuJoCo._torso_ang(env))
    # cp[3] = 
    # cp
end

function hrl_eval_net(nns::Tuple{Chain, Chain}, env, (cobmean, pobmean), (cobstd, pobstd), cintervals::Int, steps::Int, episodes::Int, cdist::Float32)
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
                # c_raw_out = forward(cnn, ob, cobmean, cobstd) * cdist
                c_raw_out = cforward(cnn, ob, LyceumMuJoCo._torso_ang(env), cobmean, cobstd, sensor_span, nbins, cdist)
                rel_target = outer_clamp.(c_raw_out, -sqrthalf, sqrthalf)
                # rel_target = outer_clamp.(rand(Uniform(-cdist, cdist), 2), -sqrthalf, sqrthalf)
                abs_target = rel_target + pos
                rewarded_prox = false
                push!(cobs, ob)

                targ_start_dist = d_old = HrlMuJoCoEnvs.euclidean(abs_target, pos)
            end

            rel_target = abs_target - pos  # update rel_target each time
            # pob = vcat(rel_target, ob)
            pob = encode_prim_obs(ob, env, rel_target, d_old / 1000)
			act = forward(pnn, pob, pobmean, pobstd)
			LyceumMuJoCo.setaction!(env, act)
			LyceumMuJoCo.step!(env)

			step += 1
			push!(pobs, pob)  # propogate ob recording to here, don't have to alloc mem if not using obs
			
            # calculating rewards
            pos = HrlMuJoCoEnvs._torso_xy(env)
            d_new = HrlMuJoCoEnvs.euclidean(pos, abs_target)

            cr += LyceumMuJoCo.getreward(env)  # TODO: ant maze only cares about last reward
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
	end
	# @show rew step
	(cr / episodes, pr / episodes), step, (cobs, pobs)
end

function cforward(nn, x, yaw, obmean, obstd, sensor_span, nbins, max_dist; rng=Random.GLOBAL_RNG)
    out = ScalableES.forward(nn, x, obmean, obstd; rng=rng)

    bin_idx = argmax(out)
    dist_percent = out[bin_idx]

    bin_res = sensor_span / nbins
    half_span = sensor_span / 2
    n_bins_inv = 1 / nbins

    angle = bin_idx * bin_res - half_span + n_bins_inv + yaw
    return pol2cart(max(dist_percent * max_dist, 1), angle)
end

end  # module