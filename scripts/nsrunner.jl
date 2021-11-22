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

    t = now()

    (cnn, pnn), obstat = make_nns(obssize, pobsize, coutsize, actsize, conf.training.pop_size)
    println("nns created")

    ScalableHrlEs.run_hrl_nses(conf.name, cnn, pnn, envs, ScalableES.ThreadComm();
                                obstat=obstat,
                                gens=conf.training.generations, 
                                interval=conf.hrl.interval, 
                                episodes=conf.training.episodes,
                                npolicies=conf.training.policies,
                                cdist=conf.hrl.cdist,
                                onehot=conf.hrl.onehot,
                                prim_specific_obs=conf.hrl.prim_specific_obs,
                                behv_freq=conf.training.behv_freq,
                                min_w=conf.training.min_nov_w)
                        
    println("Total time: $(now() - t)")
    println("Finalized!")
end

function make_nns(cobssize, pobssize, coutsize, actsize, pop_size)
    cnns = [Chain(Dense(cobssize, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, coutsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal)) for _ in 1:pop_size]

    cobstat = ScalableES.Obstat(cobssize, 1f-2)


    pnn = Chain(Dense(pobssize, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal),
            Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
            Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
            Dense(256, actsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
            x -> x .* 30)  # because ant has a joint range from -30, 30

    pobstat = ScalableES.Obstat(pobssize, 1f-2)
    

    (cnns, pnn), ScalableHrlEs.HrlObstat(cobstat, pobstat)
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

args = parseargs()
run(ScalableHrlEs.loadconfig(args["cfgpath"]), args["mjpath"])