

"""
Makes sure v is outside of the range (lower, upper) 
by rounding it to the nearest value outside the range
"""
function outer_clamp(v::T, lower::T, upper::T) where {T<:Number}
    if lower < v < upper
        delta_lower = v - lower
        delta_upper = upper - v

        v = delta_lower < delta_upper ? lower : upper
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
