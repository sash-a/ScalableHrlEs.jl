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

function run(conf, mjpath)
    println("Run name: $(conf.name)")
    savedfolder = joinpath(@__DIR__, "..", "saved", conf.name)
    if !isdir(savedfolder)
        mkdir(savedfolder)
    end

    mj_activate(mjpath)
    println("MuJoCo activated")
    println("n threads $(Threads.nthreads())")

    seed = 4321  # auto generate and share this?
    envs = HrlMuJoCoEnvs.tconstruct(nametoenv(conf.env.name)..., Threads.nthreads(); seed=seed)
    # envs = HrlMuJoCoEnvs.tconstruct(HrlMuJoCoEnvs.MazeEnv, HrlMuJoCoEnvs.PointMass, Threads.nthreads(); seed=seed)

    env = first(envs)
    @show env
    actsize::Int = length(actionspace(env))
    obssize::Int = length(obsspace(env))
    coutsize = conf.hrl.onehot ? 8 : 2

    cnn = Chain(Dense(obssize, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, coutsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal))

    pnn = Chain(Dense(obssize + 3, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, actsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                x -> x .* 30)  # because easy ant has a joint range from -30, 30
    println("nns created")
    t = now()

    ScalableHrlEs.run_hrles(conf.name, cnn, pnn, envs, ScalableES.ThreadComm(); 
                            gens=conf.training.generations, 
                            interval=conf.hrl.interval, 
                            episodes=conf.training.episodes,
                            npolicies=conf.training.policies,
                            cdist=conf.hrl.cdist,
                            pretrained_path=conf.hrl.pretrained,
                            onehot=conf.hrl.onehot)
                        
    println("Total time: $(now() - t)")
    println("Finalized!")
end

function parseargs()
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
    parse_args(s)
end

function nametoenv(name::String)
    robot = nothing
    env = nothing
    if occursin("Ant", name)
        robot = HrlMuJoCoEnvs.Ant
    elseif occursin("Point", name)
        robot = HrlMuJoCoEnvs.PointMass
    end

    if occursin("Maze", name)
        env = HrlMuJoCoEnvs.MazeEnv
    elseif occursin("Gather", name)
        env = HrlMuJoCoEnvs.GatherEnv
    end
    
    env, robot
end