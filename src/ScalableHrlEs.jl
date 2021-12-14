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
using BSON
using YAML
using Configurations
using Dates

include("policy.jl")
include("util.jl")
include("result.jl")
include("config.jl")
include("optimizer.jl")
include("obstat.jl")
include("env.jl")

function run_hrles(
    name::String,
    cnn,
    pnn,
    envs,
    comm::Union{Comm,ScalableES.ThreadComm};
    obstat = nothing,
    gens = 150,
    npolicies = 256,
    interval = 200,
    steps = 1000,
    episodes = 3,
    cdist = 4,
    σ = 0.02f0,
    nt_size = 250000000,
    η = 0.01f0,
    ctrl_pretrained_path = "",
    prim_pretrained_path = "",
    onehot = false,
    prim_specific_obs = false,
)
    @assert npolicies / ScalableES.nnodes(comm) % 2 == 0 "Num policies / num nodes must be even (eps:$npolicies, nprocs:$(ScalableES.nnodes(comm)))"

    println("Running ScalableEs")
    tblg =
        ScalableES.isroot(comm) ? ScalableES.TBLogger("tensorboard_logs/$(name)", min_level = ScalableES.Logging.Info) :
        nothing

    rngs = ScalableES.parallel_rngs(123, ScalableES.nprocs(comm), comm)

    env = first(envs)
    obssize = length(ScalableES.obsspace(env))

    println("Creating policy")
    p = HrlPolicy(cnn, pnn)
    ScalableES.bcast_policy!(p, comm)

    println("Creating noise table")
    nt, win = NoiseTable(nt_size, length(p.cπ.θ), σ, comm)

    obstat = obstat === nothing ? HrlObstat(obssize, obssize + 3, 1.0f-2) : obstat

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
    f =
        (nns, e, rng, obmean, obstd) -> hrl_run_env(
            nns,
            e,
            rng,
            obmean,
            obstd,
            interval,
            steps,
            episodes,
            cdist;
            cforward = cforward,
            prim_specific_obs = prim_specific_obs,
        )
    eval_gather = env isa HrlMuJoCoEnvs.AbstractGatherEnv
    eval_maze =
        env isa HrlMuJoCoEnvs.AbstractMazeEnv ||
        env isa HrlMuJoCoEnvs.AbstractPushEnv ||
        env isa HrlMuJoCoEnvs.AbstractFallEnv
    evalfn =
        (nns, e, rng, obmean, obstd) -> first(
            first(
                hrl_run_env(
                    nns,
                    e,
                    rng,
                    obmean,
                    obstd,
                    interval,
                    steps,
                    10,
                    cdist;
                    cforward = cforward,
                    maze_eval = eval_maze,
                    earlystop_eval = eval_gather,
                    prim_specific_obs = prim_specific_obs,
                ),
            ),
        )

    # warmup
    f(ScalableES.to_nn(p), first(envs), first(rngs), mean(obstat), std(obstat))

    println("Initialization done")
    ScalableES.run_gens(gens, name, p, nt, f, evalfn, envs, npolicies, opt, obstat, tblg, rngs, comm)

    if ScalableES.isroot(comm)
        model = ScalableES.to_nn(p)
        ScalableES.@save joinpath(@__DIR__, "..", "saved", name, "model-obstat-opt-final.bson") model obstat opt
    end

    if win !== nothing
        MPI.free(win)
    end
end

@views function ScalableES.noiseify(π::HrlPolicy, nt::ScalableES.NoiseTable, rng)
    cind = rand(rng, nt, length(π.cπ.θ))
    pind = rand(rng, nt, length(π.pπ.θ))
    cpos_pol, cneg_pol, _ = ScalableES.noiseify(π.cπ, nt, cind)
    ppos_pol, pneg_pol, _ = ScalableES.noiseify(π.pπ, nt, pind)

    HrlPolicy(cpos_pol, ppos_pol), HrlPolicy(cneg_pol, pneg_pol), (cind, pind)
end

include("novelty/ScalableHrlNsEs.jl")  # todo move to its own package

end  # module
