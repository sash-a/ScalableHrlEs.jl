include("../src/ScalableHrlEs.jl")
using .ScalableHrlEs
using ScalableES

using MuJoCo
using LyceumMuJoCo
using LyceumMuJoCoViz
using HrlMuJoCoEnvs

using Flux

using BSON:@load
using StaticArrays
using ArgParse

function runsaved(runname, gen, intervals::Int, cdist::Float32)
    @load "saved/$(runname)/model-obstat-opt-gen$gen.bson" model obstat opt
    mj_activate("/home/sasha/.mujoco/mjkey.txt")

    env = HrlMuJoCoEnvs.AntGatherEnv(viz=true)

    # states = collectstates(model, env, obmean, obstd)
    obmean = ScalableHrlEs.mean(obstat)
    obstd = ScalableHrlEs.std(obstat)
    @show first(ScalableHrlEs.hrl_eval_net(model, env, obmean, obstd, intervals, 1000, 100, cdist))
    visualize(env, controller=e -> act(model, e, intervals, obmean, obstd, 1000))
end

abs_target = [0, 0]
rel_target = [0, 0]
step = 0
targ_start_dist = 0

const cdist = 4
const sqrthalf = sqrt(1/2)

function act(nns::Tuple{Chain,Chain}, env, cintervals::Int, (cobmean, pobmean), (cobstd, pobstd), steps::Int)
    global abs_target 
    global rel_target
    global step
    global targ_start_dist
    global cdist

    cnn, pnn = nns
    pos = HrlMuJoCoEnvs._torso_xy(env)

    ob = LyceumMuJoCo.getobs(env)
    if step % cintervals == 0  # step the controller
        c_raw_out = ScalableHrlEs.cforward(cnn, ob, LyceumMuJoCo._torso_ang(env), cobmean, cobstd, env.sensor_span, env.nbins, cdist)
        rel_target = ScalableHrlEs.outer_clamp.(c_raw_out, -sqrthalf, sqrthalf)
        # rel_target = outer_clamp.(rand(Uniform(-cdist, cdist), 2), -sqrthalf, sqrthalf)
        abs_target = rel_target + pos
        @show abs_target
        getsim(env).mn[:geom_pos][ngeom=:recomend_geom] = [abs_target..., 0]
        targ_start_dist = HrlMuJoCoEnvs.euclidean(abs_target, pos)
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

s = ArgParseSettings()
    @add_arg_table s begin
    "runname"
        required = true
        help = "run name in saved folder"
    "generation"
        required = true
        help = "generation number to view"
    "--intervals", "-i"
        help = "how often the controller suggests a target"
        arg_type = Int
        default = 200
    "--dist", "-d"
        help = "max distance from agent controller can recommend"
        arg_type = Float32
        default = 4f0
end
args = parse_args(s)

runsaved(args["runname"], args["generation"], args["intervals"], args["dist"])