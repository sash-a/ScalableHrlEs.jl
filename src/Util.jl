struct HrlEsResult{T} <: ScalableES.AbstractResult{T}
    cres::ScalableES.EsResult{T}
    pres::ScalableES.EsResult{T}
end
ScalableES.sumsteps(res::AbstractVector{HrlEsResult{T}}) where T = ScalableES.sumsteps(map(r->r.pres, res))  # only counting the steps of the primitive

function ScalableES.make_result(fit::Tuple{Float64, Float64}, noise_ind::Int, steps::Int)
    HrlEsResult{Float64}(ScalableES.EsResult(first(fit), noise_ind, steps), ScalableES.EsResult(last(fit), noise_ind, steps))
end

function ScalableES.rank(results::AbstractVector{HrlEsResult{T}}) where T
    cranked = ScalableES.rank(map(r->r.cres, results))
    pranked = ScalableES.rank(map(r->r.pres, results))

    map((c, p) -> HrlEsResult(c, p), cranked, pranked)
end

function StatsBase.summarystats(results::AbstractVector{HrlEsResult{T}}) where T
    StatsBase.summarystats(map(r->r.cres, results)), StatsBase.summarystats(map(r->r.pres, results))
end

function angle_encode_target(targetvec::AbstractVector{T}, torso_ang::T) where T <: AbstractFloat
    angle_to_target = atan(targetvec[2], targetvec[1]) - torso_ang
    [sin(angle_to_target), cos(angle_to_target)]
end

function outer_clamp(v::T, lower::T, upper::T) where T <: Number
    if lower < v < upper
        delta_lower = v - lower
        delta_upper = upper - v

        v = delta_lower < delta_upper ? lower :  upper
    end

    v
end

pol2cart(rho, phi) = [rho * cos(phi), rho * sin(phi)]