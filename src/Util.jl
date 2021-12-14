struct HrlEsResult{T} <: ScalableES.AbstractResult{T}
    cres::ScalableES.EsResult{T}
    pres::ScalableES.EsResult{T}
end
ScalableES.sumsteps(res::AbstractVector{HrlEsResult{T}}) where T = ScalableES.sumsteps(map(r->r.pres, res))  # only counting the steps of the primitive

function ScalableES.make_result(fit::Tuple{Float64, Float64}, noise_ind::Tuple{Int, Int}, steps::Int)
    HrlEsResult{Float64}(ScalableES.EsResult(first(fit), first(noise_ind), steps), ScalableES.EsResult(last(fit), last(noise_ind), steps))
end

function ScalableES.rank(results::AbstractVector{HrlEsResult{T}}) where T
    # need to rank separately because positive and negative perturbs are stored in different order for ctrl and prim
    # ranking controller
    cres = map(r->r.cres, results)
    cranked_fits = ScalableES.rank(map(r->r.fit, cres))
    cranked = map((r,f)->ScalableES.EsResult(f, r.ind, r.steps), cres, cranked_fits)
	cranked = map(((p1,p2,n1,n2),) ->  # controller returns positives, positive, negative, negative...
        ScalableES.EsResult(p1.fit + p2.fit - n1.fit - n2.fit, p1.ind, p1.steps + p2.steps + n1.steps + n2.steps), 
        ScalableES.partition(cranked, 4)
    )
    # ranking primitive
    pres = map(r->r.pres, results)
    pranked_fits = ScalableES.rank(map(r->r.fit, pres))
    pranked = map((r,f)->ScalableES.EsResult(f, r.ind, r.steps), pres, pranked_fits)
	pranked = map(((p1,n1,p2,n2),) ->  # primitive returns positive, negative, positive, negative...
        ScalableES.EsResult(p1.fit + p2.fit - n1.fit - n2.fit, p1.ind, p1.steps + p2.steps + n1.steps + n2.steps), 
        ScalableES.partition(pranked, 4)
    )

    map((c, p) -> HrlEsResult(c, p), cranked, pranked)  # pack results back into HRL result
end

function StatsBase.summarystats(results::AbstractVector{HrlEsResult{T}}) where T
    StatsBase.summarystats(map(r->r.cres, results)), StatsBase.summarystats(map(r->r.pres, results))
end

"""
Given a robots target and current yaw, returns the sin and cos of the angle to the target
"""
function angle_encode_target(targetvec::AbstractVector{T}, torso_ang::T) where T <: AbstractFloat
    angle_to_target = atan(targetvec[2], targetvec[1]) - torso_ang
    [sin(angle_to_target), cos(angle_to_target)]
end

"""
Makes sure v is outside of the range (lower, upper) 
by rounding it to the nearest value outside the range
"""
function outer_clamp(v::T, lower::T, upper::T) where T <: Number
    if lower < v < upper
        delta_lower = v - lower
        delta_upper = upper - v

        v = delta_lower < delta_upper ? lower :  upper
    end

    v
end

pol2cart(rho, phi) = [rho * cos(phi), rho * sin(phi)]

"""
returns the relevant (qpos and qvel) observation range for 
just the robots obs for the corresponding env
"""
function robot_obs_range(env)
    if env isa HrlMuJoCoEnvs.AntGatherEnv
        [1:29;]
    elseif env isa HrlMuJoCoEnvs.AntMazeEnv
        [4:32;]
    elseif env isa HrlMuJoCoEnvs.AntPushEnv
        [4:18; 22:35]
    elseif env isa HrlMuJoCoEnvs.AntFallEnv
        [4:18; 22:35]
    end
end
