include("../src/ScalableHrlEs.jl")
using .ScalableHrlEs: ScalableHrlEs, ScalableES

using MuJoCo
using LyceumMuJoCo
using HrlMuJoCoEnvs

using MPI
using Base.Threads

using Flux
using LinearAlgebra

using Dates
using Random
using ArgParse
using BSON

function run(conf, mjpath, mpi)
    @show mpi
    comm = if mpi
        MPI.Init()
        MPI.COMM_WORLD  # expecting this to be one per node
    else
        ScalableES.ThreadComm()
    end

    t = now()

    println("Run name: $(conf.name)")
    savedfolder = joinpath(@__DIR__, "..", "saved", conf.name)
    if ScalableES.isroot(comm) && !isdir(savedfolder)
        mkdir(savedfolder)
    end

    # otherwise BLAS competes with SHES for threads
    LinearAlgebra.BLAS.set_num_threads(1)

    mj_activate(mjpath)
    println("MuJoCo activated")
    println("n threads $(Threads.nthreads())")

    @show conf.env.name
    @show conf.env.kwargs
    envs = LyceumMuJoCo.tconstruct(HrlMuJoCoEnvs.make(conf.env.name), Threads.nthreads(); conf.env.kwargs...)
    env = first(envs)

    actsize::Int = length(actionspace(env))
    obssize::Int = length(obsspace(env))
    coutsize = conf.hrl.onehot ? 8 : 2
    pobsize = conf.hrl.prim_specific_obs ? 29 + 3 : obssize + 3

    (cnn, pnn), obstat = make_nns(obssize, pobsize, coutsize, actsize)
    println("nns created")

    ScalableHrlEs.run_hrles(
        conf.name,
        cnn,
        pnn,
        envs,
        comm;
        obstat = obstat,
        gens = conf.training.generations,
        interval = conf.hrl.interval,
        σ = conf.training.sigma,
        η = conf.training.lr,
        episodes = conf.training.episodes,
        npolicies = conf.training.policies,
        cdist = conf.hrl.cdist,
        onehot = conf.hrl.onehot,
        prim_pretrained_path = conf.hrl.pretrained_prim,
        prim_specific_obs = conf.hrl.prim_specific_obs,
        seed = conf.seed,
    )
    println("DONE: Total time: $(now() - t)")
end

function make_nns(cobssize, pobssize, coutsize, actsize)
    cnn = Chain(
        Dense(cobssize, 256, tanh; initW = Flux.glorot_normal, initb = Flux.glorot_normal),
        Dense(256, 256, tanh; initW = Flux.glorot_normal, initb = Flux.glorot_normal),
        Dense(256, 256, tanh; initW = Flux.glorot_normal, initb = Flux.glorot_normal),
        Dense(256, coutsize, tanh; initW = Flux.glorot_normal, initb = Flux.glorot_normal),
    )

    pnn = Chain(
        Dense(pobssize, 256, tanh; initW = Flux.glorot_normal, initb = Flux.glorot_normal),
        Dense(256, 256, tanh; initW = Flux.glorot_normal, initb = Flux.glorot_normal),
        Dense(256, 256, tanh; initW = Flux.glorot_normal, initb = Flux.glorot_normal),
        Dense(256, actsize, tanh; initW = Flux.glorot_normal, initb = Flux.glorot_normal),
        x -> x .* 30,
    )  # because ant has a joint range from -30, 30

    cobstat = ScalableES.Obstat(cobssize, 1.0f-2)
    pobstat = ScalableES.Obstat(pobssize, 1.0f-2)

    (cnn, pnn), ScalableHrlEs.HrlObstat(cobstat, pobstat)
end

function parseargs()
    s = ArgParseSettings()
    @add_arg_table s begin
        "cfgpath"
        required = true
        help = "path/to/cfg.yml"
        "--mpi"
        help = "use if running on multiple nodes"
        action = :store_true
        "--mjpath"
        required = false
        default = "~/.mujoco/mjkey.txt"
        help = "path/to/mujoco/mjkey.txt"
    end
    parse_args(s)
end