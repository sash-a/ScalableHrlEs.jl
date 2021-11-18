using ScalableES: Path, Vec2, SPath, Archive
include("HrlNsEsResult.jl")


function run_hrl_nses(name::String, cnns, pnn, envs, comm::Union{Comm,ScalableES.ThreadComm}; obstat = nothing,
    gens = 150, npolicies = 256, interval = 200, steps = 1000, episodes = 3, cdist = 4, σ = 0.02f0, nt_size = 250000000, η = 0.01f0,
    onehot = false, prim_specific_obs = false)
    @assert npolicies / size(comm) % 2 == 0 "Num policies / num nodes must be even (eps:$npolicies, nprocs:$(size(comm)))"

    println("Running ScalableEs")
    tblg = ScalableES.TBLogger("tensorboard_logs/$(name)", min_level = ScalableES.Logging.Info)

    env = first(envs)
    obssize = length(ScalableES.obsspace(env))

    println("Creating policy")
    ps = [HrlPolicy(cnn, pnn, comm) for cnn in cnns]
    for p in ps
        ScalableES.bcast_policy!(p, comm)  # untested for mpi
    end

    println("Creating noise table")
    nt, win = NoiseTable(nt_size, length(first(ps).cπ.θ), σ, comm)

    obstat = obstat === nothing ? HrlObstat(obssize, obssize + 3, 1f-2) : obstat

    opt = ScalableES.isroot(comm) ? HrlAdam(length(first(ps).cπ.θ), length(first(ps).pπ.θ), η) : nothing

    cforward = onehot ? onehot_forward : forward
    f = (nns, e, obmean, obstd) -> hrl_ns_run_env(nns, e, obmean, obstd, interval, steps,
                                                episodes, cdist; cforward = cforward,
                                                prim_specific_obs = prim_specific_obs)
    eval_gather = env isa HrlMuJoCoEnvs.AbstractGatherEnv
    eval_maze = env isa HrlMuJoCoEnvs.AbstractMazeEnv ||
                env isa HrlMuJoCoEnvs.AbstractPushEnv ||
                env isa HrlMuJoCoEnvs.AbstractFallEnv
    evalfn = (nns, e, obmean, obstd) -> first(first(first(hrl_ns_run_env(nns, e, obmean, obstd, interval, steps, 10, cdist;
        cforward = cforward, rng = nothing, maze_eval = eval_maze,
        earlystop_eval = eval_gather, prim_specific_obs = prim_specific_obs))))
    behvfn = (nn, e, obmean, obstd) -> last(first(first(f(nn, e, obmean, obstd))))
    w_schedule = (w, fit, best_fit, tsb_fit) -> ScalableES.weight_schedule(w, fit, best_fit, tsb_fit; min_w=0.8)

    println("Initialization done")
    ScalableES.run_gens(gens, name, ps, nt, f, behvfn, evalfn, envs, npolicies, opt, obstat, tblg, steps, episodes, w_schedule, comm)

    for (i, p) in enumerate(ps)
        model = ScalableES.to_nn(p)
        ScalableES.@save joinpath(@__DIR__, "..", "saved", name, "model-obstat-opt-final$i.bson") model obstat opt
    end

    if win !== nothing
        MPI.free(win)
    end
end

function hrl_ns_run_env(nns::Tuple{Chain,Chain}, env, (cobmean, pobmean), (cobstd, pobstd),
    cintervals::Int, steps::Int, episodes::Int, cdist::Float32;
    cforward = onehot_forward, prim_specific_obs = false,
    rng = Random.GLOBAL_RNG, earlystop_eval = false, maze_eval = false)
    cobs = Vector{Vector{Float64}}()
    pobs = Vector{Vector{Float64}}()
    cpaths = Vector{Path}()
    ppaths = Vector{Path}()

    cr = 0
    pr = 0
    step = 0

    for ep = 1:episodes
        ((ep_cr, ep_cpath), (ep_pr, ep_ppath)), ep_step, (ep_cobs, ep_pobs) =
            _ns_one_episode(nns, env, (cobmean, pobmean), (cobstd, pobstd),
                cintervals, steps, episodes, cdist; rng = rng, cforward = cforward,
                prim_specific_obs = prim_specific_obs,
                earlystop_eval = earlystop_eval, maze_eval = maze_eval)
        cr += ep_cr
        pr += ep_pr
        step += ep_step
        cobs = vcat(cobs, ep_cobs)
        pobs = vcat(pobs, ep_pobs)
        push!(cpaths, ep_cpath)
        push!(ppaths, ep_ppath)
    end
    # @show rew step
    ((cr / episodes, cpaths), (pr / episodes, ppaths)), step, (cobs, pobs)
end

function _ns_one_episode(nns::Tuple{Chain,Chain}, env, (cobmean, pobmean), (cobstd, pobstd),
    cintervals::Int, steps::Int, episodes::Int, cdist::Float32;
    behv_freq=20, cforward = onehot_forward, prim_specific_obs = false,
    rng = Random.GLOBAL_RNG, earlystop_eval = false, maze_eval = false)

    cnn, pnn = nns

    cobs = Vector{Vector{Float64}}()
    pobs = Vector{Vector{Float64}}()
    cpath = Path()
    ppath = Path()

    cr = 0
    pr = 0
    step = 0

    targ_start_dist = 0  # dist from target when controller suggests position
    d_old = 0.0f0

    sqrthalf = sqrt(1 / 2)

    pos = zeros(2)
    rel_target = zeros(2)
    abs_target = zeros(2)

    sensor_span = hasproperty(env, :sensor_span) ? env.sensor_span : 2 * π
    nbins = hasproperty(env, :nbins) ? env.nbins : 8

    earlystop_rew = 0

    LyceumMuJoCo.reset!(env)
    for i = 0:steps-1
        ob = LyceumMuJoCo.getobs(env)

        if i % cintervals == 0  # step the controller
            c_raw_out = cforward(cnn, ob, cobmean, cobstd, cdist, LyceumMuJoCo._torso_ang(env), sensor_span, nbins; rng = rng)
            rel_target = outer_clamp.(c_raw_out, -sqrthalf, sqrthalf)
            # rel_target = outer_clamp.(rand(Uniform(-cdist, cdist), 2), -sqrthalf, sqrthalf)
            abs_target = rel_target + pos
            push!(cobs, ob)

            targ_start_dist = d_old = HrlMuJoCoEnvs.euclidean(abs_target, pos)
        end

        rel_target = abs_target - pos  # update rel_target each time

        pob = ob
        if prim_specific_obs
            pob = ob[robot_obs_range(env)]
        end
        pob = encode_prim_obs(pob, env, rel_target, d_old / 1000)
        act = ScalableES.forward(pnn, pob, pobmean, pobstd; rng = rng)
        LyceumMuJoCo.setaction!(env, act)
        LyceumMuJoCo.step!(env)

        step += 1
        push!(pobs, pob)  # propogate ob recording to here, don't have to alloc mem if not using obs
        # calculating rewards
        pos = HrlMuJoCoEnvs._torso_xy(env)
        d_new = HrlMuJoCoEnvs.euclidean(pos, abs_target)

        cr += LyceumMuJoCo.getreward(env)
        pr += 1 - (d_new / targ_start_dist) + (d_new < 1 ? 1 : 0)

        d_old = d_new

        if i % behv_freq == 0
            push!(cpath, Vec2(pos...))  # could record relative recommended pos
            push!(ppath, Vec2(pos...))  # not using now, here for consistency
        end

        if earlystop_eval
            earlystop_rew = max(earlystop_rew, cr)
        end

        if LyceumMuJoCo.isdone(env)
            break
        end
    end

    if earlystop_eval
        cr = earlystop_rew
    elseif maze_eval
        cr = LyceumMuJoCo.geteval(env)
    end

    cpath = vcat(cpath, fill(cpath[end], (steps ÷ behv_freq) - length(cpath)))
    ppath = vcat(ppath, fill(ppath[end], (steps ÷ behv_freq) - length(ppath)))

    ((cr, cpath), (pr, ppath)), step, (cobs, pobs)
end