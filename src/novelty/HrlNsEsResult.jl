struct HrlNsEsResult{T,B,S} <: ScalableES.AbstractResult{T}
    cres::ScalableES.NsEsResult{T,B,S}
    pres::ScalableES.NsEsResult{T,B,S}
end
function HrlNsEsResult(
    cres::ScalableES.NsEsResult{T,B,S},
    pres::ScalableES.NsEsResult{T,B,S},
) where {T} where {B} where {S}
    HrlNsEsResult{T,B,S}(cres, pres)
end

const NsFitType = Tuple{Tuple{Float64,Vector{Path}},Tuple{Float64,Vector{Path}}}
function ScalableES.make_result(fit::NsFitType, noise_ind::Tuple{Int,Int}, steps::Int)
    HrlNsEsResult(
        ScalableES.make_result(first(fit), first(noise_ind), steps),
        ScalableES.make_result(last(fit), last(noise_ind), steps),
    )
end

function ScalableES.make_result_vec(
    n::Int,
    ::HrlPolicy,
    rollouts::Int,
    steps::Int,
    interval::Int,
    ::ScalableES.AbstractComm,
)
    npoints = interval < 0 ? 1 : steps ÷ interval
    Vector{HrlNsEsResult{Float64,rollouts,npoints}}(undef, n)
end

function ScalableES.rank(rs::AbstractVector{T}, w) where {T<:HrlNsEsResult}
    cranked = ScalableES.rank(map(r -> r.cres, rs), convert(Float32, w))
    pranked = ScalableES.rank(map(r -> r.pres, rs), 1.0f0)  # don't want any weighting towards novelty for primitive

    map((c, p) -> HrlEsResult(c, p), cranked, pranked)
end

function ScalableES.sumsteps(res::AbstractVector{T}) where {T<:HrlNsEsResult}
    ScalableES.sumsteps(map(r -> r.pres, res))  # only counting the steps of the primitive
end

ScalableES.novelty(result::HrlNsEsResult, archive::Archive, n::Int) =
    ScalableES.novelty(result.cres.behaviours, map(a -> a.behaviour, archive), n)
function ScalableES.novelty(results::AbstractVector{T}, archive::Archive, n::Int) where {T<:HrlNsEsResult}
    # no need to calculate novelty for primitive
    map(
        r -> HrlNsEsResult(
            ScalableES.NsEsResult(r.cres.behaviours, ScalableES.novelty(r.cres, archive, n), r.cres.result),
            r.pres,
        ),
        results,
    )
end

ScalableES.performance(rs::AbstractVector{T}) where {T<:HrlNsEsResult} = mean(map(r -> r.cres.result.fit, rs))

function ScalableES.loginfo(
    tblogger,
    main_fit,
    rs::AbstractVector{T},
    tot_steps::Int,
    start_time,
    w,
    tsb_fit,
) where {T<:HrlNsEsResult}

    fitstats =
        (StatsBase.summarystats(map(r -> r.cres.result, rs)), StatsBase.summarystats(map(r -> r.pres.result, rs)))
    novstats = StatsBase.summarystats(map(r -> r.cres.novelty, rs))

    println("Main fit: $main_fit")
    println("Time since best fit: $tsb_fit")
    println("Fitness weight: $w")
    println("Total steps: $tot_steps")
    println("Time: $(ScalableES.now() - start_time)")
    println("Fitness stats:\n$fitstats")
    println("Novelty stats:\n$novstats")
    println("---------------------------------------------")

    ScalableES.with_logger(tblogger) do
        @info "" main_fitness = main_fit log_step_increment = 0
        @info "" tsb_fit = tsb_fit log_step_increment = 0
        @info "" fit_w = w log_step_increment = 0
        @info "" fitstats = fitstats log_step_increment = 0
        @info "" summarystat = fitstats log_step_increment = 0
        @info "" novstats = novstats log_step_increment = 0
        @info "" total_steps = tot_steps log_step_increment = 1
    end
end
