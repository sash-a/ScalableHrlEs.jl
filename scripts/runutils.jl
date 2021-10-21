include("../src/ScalableHrlEs.jl")
using .ScalableHrlEs: ScalableHrlEs, ScalableES

using MuJoCo
using LyceumMuJoCo
using HrlMuJoCoEnvs

using Base.Threads
using Flux
using Dates
using Random
using ArgParse
using BSON
using StaticArrays
using SharedArrays
using Distributed

function run(conf, mjpath)
    println("Run name: $(conf.name)")
    savedfolder = joinpath(@__DIR__, "..", "saved", conf.name)
    if !isdir(savedfolder)
        mkdir(savedfolder)
    end

    mj_activate(mjpath)
    println("MuJoCo activated")
    println("n threads $(Threads.nthreads())")

    @show conf.env.name
    @show conf.env.kwargs
    envs = LyceumMuJoCo.tconstruct(HrlMuJoCoEnvs.make(conf.env.name), Threads.nthreads(); conf.env.kwargs...)
    env = first(envs)
    @show env
    actsize::Int = length(actionspace(env))
    obssize::Int = length(obsspace(env))
    coutsize = conf.hrl.onehot ? 8 : 2
    pobsize = conf.hrl.prim_specific_obs ? 29 + 3 : obssize + 3
    # cnn = Chain(Dense(obssize, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal),
    #             Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
    #             Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
    #             Dense(256, coutsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal))

    # pnn = Chain(Dense(obssize + 3, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal),
    #             Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
    #             Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
    #             Dense(256, actsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
    #             x -> x .* 30)  # because ant has a joint range from -30, 30
    println("nns created")
    t = now()

    (cnn, pnn), obstat = make_nns(obssize, pobsize, coutsize, actsize; 
                                  ctrl_pretrained_path=conf.hrl.pretrained_ctrl,
                                  prim_pretrained_path="")

    ScalableHrlEs.run_hrles(conf.name, cnn, pnn, envs, ScalableES.ThreadComm();
                            obstat=obstat,
                            gens=conf.training.generations, 
                            interval=conf.hrl.interval, 
                            episodes=conf.training.episodes,
                            npolicies=conf.training.policies,
                            cdist=conf.hrl.cdist,
                            onehot=conf.hrl.onehot,
                            prim_pretrained_path=conf.hrl.pretrained_prim,
                            prim_specific_obs=conf.hrl.prim_specific_obs)
                        
    println("Total time: $(now() - t)")
    println("Finalized!")
end

function make_nns(cobssize, pobssize, coutsize, actsize;ctrl_pretrained_path="", prim_pretrained_path="")
    cnn, cobstat, pnn, pobstat = nothing, nothing, nothing, nothing

    if !isempty(ctrl_pretrained_path)
        loaded = BSON.load(ctrl_pretrained_path, @__MODULE__)
        cnn = ScalableES.to_nn(loaded[:p].cπ)
        cobstat = loaded[:obstat].cobstat
    else
        cnn = Chain(Dense(cobssize, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal),
        Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
        Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
        Dense(256, coutsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal))

        cobstat = ScalableES.Obstat(cobssize, 1f-2)
    end

    if !isempty(prim_pretrained_path)
        loaded = BSON.load(prim_pretrained_path, @__MODULE__)
        # pnn = ScalableES.to_nn(loaded[:p].pπ)
        pnn = ScalableES.to_nn(loaded[:p])
        pobstat = loaded[:obstat].pobstat
    else
        pnn = Chain(Dense(pobssize, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, actsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                x -> x .* 30)  # because ant has a joint range from -30, 30

        pobstat = ScalableES.Obstat(pobssize, 1f-2)
    end

    (cnn, pnn), ScalableHrlEs.HrlObstat(cobstat, pobstat)
end

function parseargs()
    s = ArgParseSettings()
    @add_arg_table s begin
        "cfgpath"
            required = true
            help = "path/to/cfg.yml"
        "--mjpath"
            required = false
            default = "~/.mujoco/mjkey.txt"
            help = "path/to/mujoco/mjkey.txt"
    end
    parse_args(s)
end