function hrl_run_env(
    nns::Tuple{Chain,Chain},
    env,
    rng,
    (cobmean, pobmean),
    (cobstd, pobstd),
    cintervals::Int,
    steps::Int,
    episodes::Int,
    cdist::Float32;
    cforward = onehot_forward,
    prim_specific_obs = false,
    earlystop_eval = false,
    maze_eval = nothing,
)
    cobs = Vector{Vector{Float64}}()
    pobs = Vector{Vector{Float64}}()

    cr = 0
    pr = 0
    step = 0

    for ep = 1:episodes
        (ep_cr, ep_pr), ep_step, (ep_cobs, ep_pobs) = _one_episode(
            nns,
            env,
            rng,
            (cobmean, pobmean),
            (cobstd, pobstd),
            cintervals,
            steps,
            episodes,
            cdist;
            cforward = cforward,
            prim_specific_obs = prim_specific_obs,
            earlystop_eval = earlystop_eval,
            maze_eval = maze_eval,
        )
        cr += ep_cr
        pr += ep_pr
        step += ep_step
        cobs = vcat(cobs, ep_cobs)
        pobs = vcat(pobs, ep_pobs)
    end
    # @show rew step
    (cr / episodes, pr / episodes), step, (cobs, pobs)
end

function _one_episode(
    nns::Tuple{Chain,Chain},
    env,
    rng,
    (cobmean, pobmean),
    (cobstd, pobstd),
    cintervals::Int,
    steps::Int,
    episodes::Int,
    cdist::Float32;
    cforward = onehot_forward,
    prim_specific_obs = false,
    earlystop_eval = false,
    maze_eval = nothing,
)
    cnn, pnn = nns

    cobs = Vector{Vector{Float64}}()
    pobs = Vector{Vector{Float64}}()

    cr = 0
    pr = 0
    step = 0

    targ_start_dist = 0  # dist from target when controller suggests position
    d_old = 0.0f0

    sqrthalf = Float32(sqrt(1 / 2))

    pos = zeros(2)
    rel_target = zeros(2)
    abs_target = zeros(2)

    sensor_span = Float32(hasproperty(env, :sensor_span) ? env.sensor_span : 2 * π)
    nbins = hasproperty(env, :nbins) ? env.nbins : 8

    earlystop_rew = 0

    LyceumMuJoCo.reset!(env)
    if maze_eval !== nothing
        env.target = maze_eval
    end

    for i = 0:steps-1
        ob = LyceumMuJoCo.getobs(env)

        if i % cintervals == 0  # step the controller
            c_raw_out = cforward(cnn, ob, cobmean, cobstd, cdist, LyceumMuJoCo._torso_ang(env), sensor_span, nbins, rng)
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

        act = ScalableES.forward(pnn, pob, pobmean, pobstd, rng)
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

        if earlystop_eval
            earlystop_rew = max(earlystop_rew, cr)
        end

        if LyceumMuJoCo.isdone(env)
            break
        end
    end

    if earlystop_eval
        cr = earlystop_rew
    elseif maze_eval !== nothing
        cr = LyceumMuJoCo.geteval(env)
    end

    (cr, pr), step, (cobs, pobs)
end

function encode_prim_obs(obs::Vector{T}, env, targ_vec, targ_dist) where {T<:AbstractFloat}
    vcat(angle_encode_target(targ_vec, LyceumMuJoCo._torso_ang(env)), [targ_dist], obs)
end

"""Given a robots target and current yaw, returns the sin and cos of the angle to the target"""
function angle_encode_target(targetvec::AbstractVector{T}, torso_ang::T) where {T<:AbstractFloat}
    angle_to_target = atan(targetvec[2], targetvec[1]) - torso_ang
    [sin(angle_to_target), cos(angle_to_target)]
end
