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
using ArgParse

# visualize(env, controller=e -> act(model, e, intervals, obmean, obstd, 1000, cdist))

abs_target = [0, 0]
rel_target = [0, 0]
step = 0
targ_start_dist = 0
rew = 0

const sqrthalf = Float32(sqrt(1 / 2))

function act(
    nns::Tuple{Chain,Chain},
    env,
    cintervals::Int,
    (cobmean, pobmean),
    (cobstd, pobstd),
    steps::Int,
    cdist;
    cforward = ScalableHrlEs.forward,
    prim_specific_obs = false,
)
    global abs_target
    global rel_target
    global step
    global targ_start_dist
    global rew

    # env.target = [0, 16]

    cnn, pnn = nns
    pos = HrlMuJoCoEnvs._torso_xy(env)

    sensor_span = hasproperty(env, :sensor_span) ? env.sensor_span : 2 * π
    nbins = hasproperty(env, :nbins) ? env.nbins : 10
    if hasproperty(env, :target)
        # getsim(env).mn[:geom_pos][ngeom=:target_geom] = [env.target..., 0]
    end

    ob = LyceumMuJoCo.getobs(env)
    if step % cintervals == 0  # step the controller
        c_raw_out = cforward(cnn, ob, cobmean, cobstd, cdist, LyceumMuJoCo._torso_ang(env), sensor_span, nbins, nothing)
        rel_target = ScalableHrlEs.outer_clamp.(c_raw_out, -sqrthalf, sqrthalf)
        abs_target = rel_target + pos
        # getsim(env).mn[:geom_pos][ngeom=:recomend_geom] = [abs_target..., 0]

        dist = HrlMuJoCoEnvs.euclidean(env.target, HrlMuJoCoEnvs._torso_xy(env))
        @show step
        @show env.start_targ_dist
        @show dist / env.start_targ_dist
        @show dist
        @show rew
        println("\n\n")
    end

    rel_target = abs_target - pos  # update rel_target each time
    dist = HrlMuJoCoEnvs.euclidean(HrlMuJoCoEnvs._torso_xy(env), abs_target)


    pob = ob
    if prim_specific_obs
        @show :PSO
        pob = ob[ScalableHrlEs.robot_obs_range(env)]
    end

    pob = ScalableHrlEs.encode_prim_obs(pob, env, rel_target, dist / 1000)
    act = ScalableES.forward(pnn, pob, pobmean, pobstd, nothing)
    LyceumMuJoCo.setaction!(env, act)
    rew += getreward(env)

    step += 1

    if LyceumMuJoCo.isdone(env)
        println("done")
    end
end


function evalenv_withtarg(
    nns::Tuple{Chain,Chain},
    env,
    (cobmean, pobmean),
    (cobstd, pobstd),
    targ,
    targdist,
    cintervals::Int,
    steps::Int,
    episodes::Int,
    cdist::Float32;
    cforward = ScalableHrlEs.forward,
)
    cnn, pnn = nns

    cobs = Vector{Vector{Float64}}()
    pobs = Vector{Vector{Float64}}()

    cr = 0
    pr = 0
    step = 0

    rewarded_prox = false  # rewarded primitive for being close to target
    sqrthalf = sqrt(1 / 2)

    max_rs = 0

    sensor_span = hasproperty(env, :sensor_span) ? env.sensor_span : 2 * π
    nbins = hasproperty(env, :nbins) ? env.nbins : 8

    for ep = 1:episodes
        LyceumMuJoCo.reset!(env)
        env.target = targ

        died = false

        pos = zeros(2)
        rel_target = zeros(2)
        abs_target = zeros(2)

        targ_start_dist = 0  # dist from target when controller suggests position
        d_old = 0.0f0

        for i = 0:steps-1
            ob = LyceumMuJoCo.getobs(env)

            if i % cintervals == 0  # step the controller
                c_raw_out = cforward(
                    cnn,
                    ob,
                    cobmean,
                    cobstd,
                    cdist,
                    LyceumMuJoCo._torso_ang(env),
                    sensor_span,
                    nbins;
                    rng = nothing,
                )
                rel_target = ScalableHrlEs.outer_clamp.(c_raw_out, -sqrthalf, sqrthalf)
                # rel_target = outer_clamp.(rand(Uniform(-cdist, cdist), 2), -sqrthalf, sqrthalf)
                abs_target = rel_target + pos
                rewarded_prox = false
                push!(cobs, ob)

                targ_start_dist = d_old = HrlMuJoCoEnvs.euclidean(abs_target, pos)
            end

            rel_target = abs_target - pos  # update rel_target each time
            # pob = vcat(rel_target, ob)
            pob = ScalableHrlEs.encode_prim_obs(ob, env, rel_target, d_old / 1000)
            act = ScalableES.forward(pnn, pob, pobmean, pobstd; rng = nothing)
            LyceumMuJoCo.setaction!(env, act)
            LyceumMuJoCo.step!(env)

            step += 1
            push!(pobs, pob)  # propogate ob recording to here, don't have to alloc mem if not using obs

            # calculating rewards
            pos = HrlMuJoCoEnvs._torso_xy(env)
            d_new = HrlMuJoCoEnvs.euclidean(pos, abs_target)

            cr += LyceumMuJoCo.getreward(env)
            # pr += (d_old - d_new) / LyceumMuJoCo.timestep(env)
            # if d_new < 1^2 && !rewarded_prox
            #     rewarded_prox = true
            # end
            pr += 1 - (d_new / targ_start_dist)
            pr += d_new < 1 ? 1 : 0

            d_old = d_new

            if LyceumMuJoCo.isdone(env)
                died = true
                break
            end
        end
        if HrlMuJoCoEnvs.euclidean(HrlMuJoCoEnvs._torso_xy(env), targ) < targdist
            max_rs += 1
            @show :reached
        end
        # max_rs += max_r
    end
    # @show rew step
    max_rs / episodes, (cr / episodes, pr / episodes)
end

# s = ArgParseSettings()
#     @add_arg_table s begin
#     "runname"
#         required = true
#         help = "run name in saved folder"
#     "env"
#         required = true
#         help = "HrlMuJoCoEnvs environment name"
#     "generation"
#         required = true
#         help = "generation number to view"
#     "--intervals", "-i"
#         help = "how often the controller suggests a target"
#         arg_type = Int
#         default = 25
#     "--dist", "-d"
#         help = "max distance from agent controller can recommend"
#         arg_type = Float32
#         default = 4f0
#     "--onehot"
#         help = "if provided then controller uses one hot mode"
#         action = :store_true
# end
# args = parse_args(s)

# runname = args["runname"]
# envname = args["env"]
# gen = args["generation"]
# intervals = args["intervals"]
# cdist = args["dist"]
# cforward = args["onehot"] ? ScalableHrlEs.onehot_forward : ScalableHrlEs.forward

envname = "AntMaze"
runname = "remote/maze/AntMaze-pretrained_pretrained_1"
gen = 1

intervals = 25
cdist = 4

mj_activate("/home/sasha/.mujoco/mjkey.txt")
env = first(LyceumMuJoCo.tconstruct(HrlMuJoCoEnvs.make(envname), 1))

@show actionspace(env)
@show obsspace(env)

@load "saved/$(runname)/policy-obstat-opt-gen$gen.bson" p obstat opt
obmean = ScalableHrlEs.mean(obstat)
obstd = ScalableHrlEs.std(obstat)
model = Base.invokelatest(ScalableES.to_nn, p)

# @show evalenv_withtarg(model, env, obmean, obstd, HrlMuJoCoEnvs.MAZE_TARGET, 5, intervals, 500, 10, cdist; cforward=cforward)
visualize(env, controller = e -> act(model, e, intervals, obmean, obstd, 500, cdist; prim_specific_obs=true))
