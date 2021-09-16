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


function run(cfgpath, mjpath)
    conf = ScalableHrlEs.loadconfig(cfgpath)

    println("Run name: $(conf.name)")
    savedfolder = joinpath(@__DIR__, "..", "saved", conf.name)
    if !isdir(savedfolder)
        mkdir(savedfolder)
    end

    mj_activate(mjpath)
    println("MuJoCo activated")

    println("n threads $(Threads.nthreads())")
    

    seed = 4321  # auto generate and share this?
    # envs = HrlMuJoCoEnvs.tconstruct(HrlMuJoCoEnvs.AntGather, Threads.nthreads(); seed=seed)
    envs = HrlMuJoCoEnvs.tconstruct(HrlMuJoCoEnvs.PointGatherEnv, Threads.nthreads(); seed=seed)

    env = first(envs)
    actsize::Int = length(actionspace(env))
    obssize::Int = length(obsspace(env))
    coutsize = 10

    cnn = Chain(Dense(obssize, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, coutsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal))

    pnn = Chain(Dense(obssize + 3, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, actsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal))
    println("nns created")
    t = now()

    ScalableHrlEs.run_hrles(conf.name, cnn, pnn, envs, ScalableES.ThreadComm(); 
                            gens=conf.training.generations, 
                            interval=conf.hrl.interval, 
                            episodes=conf.training.episodes,
                            npolicies=conf.training.policies,
                            cdist=conf.hrl.cdist,
                            pretrained_path=conf.hrl.pretrained)
                        
    println("Total time: $(now() - t)")
    println("Finalized!")
end

s = ArgParseSettings()
@add_arg_table s begin
    "cfgpath"
        required=true
        help="path/to/cfg.yml"
    "--mjpath"
        required=false
        default="~/.mujoco/mjkey.txt"
        help="path/to/mujoco/mjkey.txt"
end
args = parse_args(s)

run(args["cfgpath"], args["mjpath"])