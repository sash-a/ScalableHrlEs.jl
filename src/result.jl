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

ScalableES.make_result_vec(n::Int64, ::HrlPolicy{Float32}, ::ScalableES.AbstractComm) =
    Vector{HrlEsResult{Float64}}(undef, n)

function ScalableES.rank(results::AbstractVector{HrlEsResult{T}}) where {T}
    # need to rank separately because positive and negative perturbs are stored in different order for ctrl and prim
    # ranking controller
    cres = map(r -> r.cres, results)
    cranked_fits = ScalableES.rank(map(r -> r.fit, cres))
    cranked = map((r, f) -> ScalableES.EsResult(f, r.ind, r.steps), cres, cranked_fits)
    cranked = map(
        ((p1, p2, n1, n2),) ->  # controller returns positives, positive, negative, negative...
            ScalableES.EsResult(p1.fit + p2.fit - n1.fit - n2.fit, p1.ind, p1.steps + p2.steps + n1.steps + n2.steps),
        ScalableES.partition(cranked, 4),
    )
    # ranking primitive
    pres = map(r -> r.pres, results)
    pranked_fits = ScalableES.rank(map(r -> r.fit, pres))
    pranked = map((r, f) -> ScalableES.EsResult(f, r.ind, r.steps), pres, pranked_fits)
    pranked = map(
        ((p1, n1, p2, n2),) ->  # primitive returns positive, negative, positive, negative...
            ScalableES.EsResult(p1.fit + p2.fit - n1.fit - n2.fit, p1.ind, p1.steps + p2.steps + n1.steps + n2.steps),
        ScalableES.partition(pranked, 4),
    )

    map((c, p) -> HrlEsResult(c, p), cranked, pranked)  # pack results back into HRL result
end

ScalableES.sumsteps(res::AbstractVector{HrlEsResult{T}}) where {T} = ScalableES.sumsteps(map(r -> r.pres, res))  # only counting the steps of the primitive

function StatsBase.summarystats(results::AbstractVector{HrlEsResult{T}}) where {T}
    StatsBase.summarystats(map(r -> r.cres, results)), StatsBase.summarystats(map(r -> r.pres, results))
end
