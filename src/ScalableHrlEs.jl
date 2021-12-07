module ScalableHrlEs
# Author Sasha Abramowitz
# prefixing variables with 'p' or 'c' refers to the primitive or controller version of that variable

using HrlMuJoCoEnvs

import ScalableES
using ScalableES: NoiseTable, LyceumMuJoCo, Policy

using MPI: MPI, Comm
using Flux
import StatsBase
import Statistics: mean, std
using Distributions
using Random
using SharedArrays

using StaticArrays
using SharedArrays
using Distributed
using BSON
using YAML
using Configurations

include("Util.jl")
include("Config.jl")
include("Policy.jl")
include("Optim.jl")
include("Obstat.jl")

function run_hrles(name::String, cnn, pnn, envs, comm::Union{Comm, ScalableES.ThreadComm}; obstat=nothing,
                   gens=150, npolicies=256, interval=200, steps=1000, episodes=3, cdist=4, σ=0.02f0, nt_size=250000000, η=0.01f0, 
                   ctrl_pretrained_path="", prim_pretrained_path="", onehot=false, prim_specific_obs=false)
    @assert npolicies / size(comm) % 2 == 0 "Num policies / num nodes must be even (eps:$npolicies, nprocs:$(size(comm)))"

    println("Running ScalableEs")
    tblg = nothing
    if ScalableES.isroot(comm)
        tblg = ScalableES.TBLogger("tensorboard_logs/$(name)", min_level=ScalableES.Logging.Info)
    end
    
    env = first(envs)
    obssize = length(ScalableES.obsspace(env))

    println("Creating policy")
    p = HrlPolicy(cnn, pnn, comm)
    ScalableES.bcast_policy!(p, comm)

    println("Creating noise table")
    nt, win = NoiseTable(nt_size, length(p.cπ.θ), σ, comm)

    obstat = obstat === nothing ? HrlObstat(obssize, obssize + 3, 1f-2) : obstat
    
    if !isempty(prim_pretrained_path)
        loaded = BSON.load(prim_pretrained_path, @__MODULE__)
        # pnn = ScalableES.to_nn(loaded[:p].pπ)
        p.pπ.θ = loaded[:p].θ
        @show length(p.pπ.θ)
        obstat = HrlObstat(obstat.cobstat, loaded[:obstat])
        # obstat.pobstat = loaded[:obstat]
    end

    # if !isempty(ctrl_pretrained_path)
    #     loaded = BSON.load(ctrl_pretrained_path, @__MODULE__)
    #     p.cπ = loaded[:p].cπ
    #     obstat.cobstat = loaded[:obstat].cobstat
    #     p = HrlPolicy{Float32}(model...)
    # end

    opt = ScalableES.isroot(comm) ? HrlAdam(length(p.cπ.θ), length(p.pπ.θ), η) : nothing

    cforward = onehot ? onehot_forward : forward
    f = (nns, e, obmean, obstd) -> hrl_run_env(nns, e, obmean, obstd, interval, steps, 
                                                episodes, cdist; cforward=cforward, 
                                                prim_specific_obs=prim_specific_obs)
    eval_gather = env isa HrlMuJoCoEnvs.AbstractGatherEnv
    eval_maze   = env isa HrlMuJoCoEnvs.AbstractMazeEnv || 
                  env isa HrlMuJoCoEnvs.AbstractPushEnv || 
                  env isa HrlMuJoCoEnvs.AbstractFallEnv
    evalfn = (nns, e, obmean, obstd) -> first(first(hrl_run_env(nns, e, obmean, obstd, interval, steps, 10, cdist; 
                                                            cforward=cforward, rng=nothing, maze_eval=eval_maze,
                                                            earlystop_eval=eval_gather, prim_specific_obs=prim_specific_obs)))
    println("Initialization done")
    ScalableES.run_gens(gens, name, p, nt, f, evalfn, envs, npolicies, opt, obstat, tblg, comm)

    if ScalableES.isroot(comm)
        model = ScalableES.to_nn(p)
        ScalableES.@save joinpath(@__DIR__, "..", "saved", name, "model-obstat-opt-final.bson") model obstat opt
    end

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
# ScalableES.make_obstat(shape, ::HrlPolicy) = HrlObstat(shape, shape + 3, 0f0)

function ScalableES.make_obstat(shape, pol::HrlPolicy)
    (cnn, pnn) = ScalableES.to_nn(pol)
    HrlObstat(shape, last(size(first(pnn.layers).W)), 0f0)
end

function ScalableES.bcast_policy!(π::HrlPolicy, comm::Comm)
    ScalableES.bcast_policy!(π.cπ, comm)
    ScalableES.bcast_policy!(π.pπ, comm)
end
function ScalableES.bcast_policy!(::HrlPolicy, ::ScalableES.ThreadComm) end

function encode_prim_obs(obs::Vector{T}, env, targ_vec, targ_dist) where T <: AbstractFloat
    vcat(angle_encode_target(targ_vec, LyceumMuJoCo._torso_ang(env)), [targ_dist], obs)
end

function hrl_run_env(nns::Tuple{Chain, Chain}, env, (cobmean, pobmean), (cobstd, pobstd), 
                    cintervals::Int, steps::Int, episodes::Int, cdist::Float32; 
                    cforward=onehot_forward, prim_specific_obs=false,
                    rng=Random.GLOBAL_RNG, earlystop_eval=false, maze_eval=false)    
    cobs = Vector{Vector{Float64}}()
    pobs = Vector{Vector{Float64}}()

	cr = 0
    pr = 0
	step = 0

	for ep in 1:episodes
        (ep_cr, ep_pr), ep_step, (ep_cobs, ep_pobs) = _one_episode(nns, env, (cobmean, pobmean), (cobstd, pobstd), 
                    cintervals, steps, episodes, cdist; rng = rng, cforward=cforward, 
                    prim_specific_obs=prim_specific_obs, earlystop_eval=earlystop_eval, maze_eval=maze_eval)
        cr += ep_cr
        pr += ep_pr
        step += ep_step
        cobs = vcat(cobs, ep_cobs)
        pobs = vcat(pobs, ep_pobs)
	end
	# @show rew step
	(cr / episodes, pr / episodes), step, (cobs, pobs)
end

function _one_episode(nns::Tuple{Chain, Chain}, env, (cobmean, pobmean), (cobstd, pobstd), 
                      cintervals::Int, steps::Int, episodes::Int, cdist::Float32; 
                      cforward=onehot_forward, prim_specific_obs=false,
                      rng=Random.GLOBAL_RNG, earlystop_eval=false, maze_eval=false)
    
    cnn, pnn = nns

    cobs = Vector{Vector{Float64}}()
    pobs = Vector{Vector{Float64}}()

    cr = 0
    pr = 0
    step = 0
    
    targ_start_dist = 0  # dist from target when controller suggests position
    d_old = 0f0

    sqrthalf = sqrt(1/2)
    
    pos = zeros(2)
    rel_target = zeros(2)
    abs_target = zeros(2)

    sensor_span = hasproperty(env, :sensor_span) ? env.sensor_span : 2 * π
    nbins = hasproperty(env, :nbins) ? env.nbins : 8

    earlystop_rew = 0

    LyceumMuJoCo.reset!(env)
    for i in 0:steps - 1
        ob = LyceumMuJoCo.getobs(env)

        if i % cintervals == 0  # step the controller
            c_raw_out = cforward(cnn, ob, cobmean, cobstd, cdist, LyceumMuJoCo._torso_ang(env), sensor_span, nbins; rng=rng)
            rel_target = outer_clamp.(c_raw_out, -sqrthalf, sqrthalf)
            # rel_target = outer_clamp.(rand(Uniform(-cdist, cdist), 2), -sqrthalf, sqrthalf)
            abs_target = rel_target + pos
            push!(cobs, ob)

            targ_start_dist = d_old = HrlMuJoCoEnvs.euclidean(abs_target, pos)
        end

        rel_target = abs_target - pos  # update rel_target each time

        pob = ob
        if prim_specific_obs
            pob = ob[robot_obs_range(env)]
        end
        pob = encode_prim_obs(pob, env, rel_target, d_old / 1000)
        act = ScalableES.forward(pnn, pob, pobmean, pobstd; rng=rng)
        LyceumMuJoCo.setaction!(env, act)
        LyceumMuJoCo.step!(env)

        step += 1
        push!(pobs, pob)  # propogate ob recording to here, don't have to alloc mem if not using obs
        
        # calculating rewards
        pos = HrlMuJoCoEnvs._torso_xy(env)
        d_new = HrlMuJoCoEnvs.euclidean(pos, abs_target)

        cr += LyceumMuJoCo.getreward(env)
        pr += 1 - (d_new / targ_start_dist) + (d_new < 1 ? 1 : 0)
        
        d_old = d_new
        
        if earlystop_eval
            earlystop_rew = max(earlystop_rew, cr)
        end

        if LyceumMuJoCo.isdone(env)
            break
        end
    end

    if earlystop_eval
        cr = earlystop_rew
    elseif maze_eval
        cr = LyceumMuJoCo.geteval(env)
    end

    (cr, pr), step, (cobs, pobs)
end

# so that it matches the signature of onehot_forward
forward(nn, x, obmean, obstd, cdist, y, s, n; rng=Random.GLOBAL_RNG) = ScalableES.forward(nn, x, obmean, obstd; rng=rng) * cdist
function onehot_forward(nn, x, obmean, obstd, max_dist, yaw, sensor_span, nbins; rng=Random.GLOBAL_RNG)
    out = ScalableES.forward(nn, x, obmean, obstd; rng=rng)

    bin_idx = argmax(out)
    dist_percent = out[bin_idx]

    bin_res = sensor_span / nbins
    half_span = sensor_span / 2
    n_bins_inv = 1 / nbins

    angle = bin_idx * bin_res - half_span + n_bins_inv + yaw
    return pol2cart(max(dist_percent * max_dist, 1), angle)
end

include("novelty/ScalableHrlNsEs.jl")

end  # module
