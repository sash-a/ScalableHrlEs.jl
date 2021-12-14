struct HrlEsResult{T} <: ScalableES.AbstractResult{T}
    cres::ScalableES.EsResult{T}
    pres::ScalableES.EsResult{T}
end

function ScalableES.make_result(fit::Tuple{Float64,Float64}, noise_ind::Tuple{Int,Int}, steps::Int)
    HrlEsResult{Float64}(
        ScalableES.EsResult(first(fit), first(noise_ind), steps),
        ScalableES.EsResult(last(fit), last(noise_ind), steps),
    )
end
ScalableES.make_result_vec(n::Int, ::HrlPolicy, ::ScalableES.AbstractComm) = Vector{HrlEsResult{Float64}}(undef, n)


ScalableES.sumsteps(res::AbstractVector{HrlEsResult{T}}) where {T} = ScalableES.sumsteps(map(r -> r.pres, res))  # only counting the steps of the primitive


function ScalableES.rank(results::AbstractVector{HrlEsResult{T}}) where {T}
    cranked = ScalableES.rank(map(r -> r.cres, results))
    pranked = ScalableES.rank(map(r -> r.pres, results))

    map((c, p) -> HrlEsResult(c, p), cranked, pranked)
end

function StatsBase.summarystats(results::AbstractVector{HrlEsResult{T}}) where {T}
    StatsBase.summarystats(map(r -> r.cres, results)), StatsBase.summarystats(map(r -> r.pres, results))
end
