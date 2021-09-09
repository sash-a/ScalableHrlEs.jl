include("../src/ScalableHrlEs.jl")
using .ScalableHrlEs
using ScalableES

using MuJoCo
using LyceumMuJoCo
using LyceumMuJoCoViz
using HrlMuJoCoEnvs

using Flux

using BSON: @load
using StaticArrays

function runsaved(runname, suffix)
    @load "saved/$(runname)/model-obstat-opt-$suffix.bson" model obstat opt
    
    @show model typeof(obstat)

    mj_activate("/home/sasha/.mujoco/mjkey.txt")
    seed = rand(1:100000)
    env = HrlMuJoCoEnvs.AntMaze(seed=seed)

    @show getsim(env).mn[:geom_pos]


    # states = collectstates(model, env, obmean, obstd)
    obmean = ScalableHrlEs.mean(obstat)
    obstd = ScalableHrlEs.std(obstat)
    
    visualize(env, controller = e -> act(model, e, 200, obmean, obstd, 1000))
end

abs_target = [0, 0]
rel_target = [0, 0]
step = 0

function act(nns::Tuple{Chain, Chain}, env, cintervals::Int, (cobmean, pobmean), (cobstd, pobstd), steps::Int)
    global abs_target 
    global rel_target
    global step

    cnn, pnn = nns
    pos = HrlMuJoCoEnvs._torso_xy(env)

    ob = LyceumMuJoCo.getobs(env)
    if step % cintervals == 0  # step the controller
        rel_target = ScalableES.forward(cnn, ob, cobmean, cobstd) * 5                  
        abs_target = rel_target + pos
        @show abs_target
        getsim(env).mn[:geom_pos][7:8] = abs_target
    end

    rel_target = abs_target - pos  # update rel_target each time
    dist = HrlMuJoCoEnvs.sqeuclidean(HrlMuJoCoEnvs._torso_xy(env), abs_target)
    pob = ScalableHrlEs.encode_prim_obs(ob, env, rel_target, dist)
    act = ScalableES.forward(pnn, pob, pobmean, pobstd)
    LyceumMuJoCo.setaction!(env, act)

    step += 1

    if LyceumMuJoCo.isdone(env)
        println("done")
    end
end

runsaved("angleobreplacement-doneontarg", "gen1500")