module ScalableHrlEs
# Author Sasha Abramowitz
# prefixing variables with 'p' or 'c' refers to the primitive or controller version of that variable

import ScalableES
using ScalableES: NoiseTable, LyceumMuJoCo, Policy

using HrlMuJoCoEnvs

using MPI: MPI, Comm
using ThreadPools

using Flux
import StatsBase
import Statistics: mean, std
using Distributions
using Random

using StaticArrays
using BSON
using YAML
using Configurations

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
    seed = 123
)
    validateparams(npolicies, comm)

    println("Running ScalableEs")
    tblg = if ScalableES.isroot(comm)
        ScalableES.TBLogger("tensorboard_logs/$(name)", min_level = ScalableES.Logging.Info)
    else
        nothing
    end

    rngs = ScalableES.parallel_rngs(seed, ScalableES.nprocs(comm), comm)

    env = first(envs)
    obssize = length(ScalableES.obsspace(env))

    println("Creating policy")
    p = HrlPolicy(cnn, pnn)
    ScalableES.bcast_policy!(p, comm)

    println("Creating noise table")
    nt, win = NoiseTable(nt_size, length(p.cπ.θ), σ, comm; seed=seed)

    obstat = obstat === nothing ? HrlObstat(obssize, obssize + 3, 1.0f-2) : obstat

    if !isempty(prim_pretrained_path)
        loaded = BSON.load(prim_pretrained_path, @__MODULE__)
        p.pπ.θ = loaded[:p].θ
        obstat = HrlObstat(obstat.cobstat, loaded[:obstat])

        println("Loaded pretrianed primitive ($prim_pretrained_path)")
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
                    nothing,
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

"""
Runs the evaluate loop for SHES: perturb, run env, store results

    Each policy is perturbed four times and results are stored as follows:
    controller: [pos perturb, pos perturb, neg perturb, neg perturb, ...]
    primitive: [pos perturb, neg perturb, pos perturb, neg perturb, ...]

    `results` and `obstat` are empty containers of the correct type
"""
function ScalableES.evaluate(pol::HrlPolicy, nt, f, envs, n::Int, results, obstat, rngs, comm::ScalableES.AbstractComm)
    l = ReentrantLock()

    @qthreads for i = 1:(n÷4)  # ÷ 4 because doing 4 way antithetic sampling to reduce variance
        tid = Threads.threadid()
        env = envs[tid]
        rng = rngs[tid]

        ppπ, pnπ, nnπ, npπ, noise_ind = ScalableES.noiseify(pol, nt, rng)

        ppfit, ppsteps, ppobs = f(ScalableES.to_nn(ppπ), env, rng)
        nnfit, nnsteps, nnobs = f(ScalableES.to_nn(nnπ), env, rng)
        npfit, npsteps, npobs = f(ScalableES.to_nn(npπ), env, rng)
        pnfit, pnsteps, pnobs = f(ScalableES.to_nn(pnπ), env, rng)


        if rand() < 0.01
            Base.@lock l begin
                obstat = ScalableES.add_obs(obstat, ppobs)
                obstat = ScalableES.add_obs(obstat, nnobs)
                obstat = ScalableES.add_obs(obstat, pnobs)
                obstat = ScalableES.add_obs(obstat, npobs)
            end
        end
        results[i*4-3] = ScalableES.make_result(ppfit, noise_ind, ppsteps)
        results[i*4-2] = ScalableES.make_result(pnfit, noise_ind, pnsteps)
        results[i*4-1] = ScalableES.make_result(npfit, noise_ind, npsteps)
        results[i*4-0] = ScalableES.make_result(nnfit, noise_ind, nnsteps)
    end

    results, obstat
end

@views function ScalableES.noiseify(π::HrlPolicy, nt::ScalableES.NoiseTable, rng)
    cind = rand(rng, nt, length(π.cπ.θ))
    pind = rand(rng, nt, length(π.pπ.θ))
    cpos_pol, cneg_pol, _ = ScalableES.noiseify(π.cπ, nt, cind)
    # ppos_pol, pneg_pol, _ = ScalableES.noiseify(π.pπ, nt, pind)

    HrlPolicy(cpos_pol, π.pπ),
    HrlPolicy(cpos_pol, π.pπ),
    HrlPolicy(cneg_pol, π.pπ),
    HrlPolicy(cneg_pol, π.pπ),
    (cind, pind)
end

function validateparams(pols::Int, comm::ScalableES.AbstractComm)
    nodes = ScalableES.nnodes(comm)
    ppn = pols / nodes
    @assert pols % nodes == 0 "Each node must get the same number of policies. There are $pols policies and $nodes nodes."
    @assert ppn % 4 == 0 "Policies per node must be divisible by four to perform antithetic sampling. Policies per node = $ppn."
end

include("novelty/ScalableHrlNsEs.jl")  # todo move to its own package

end  # module
