mutable struct HrlAdam <: ScalableES.AbstractOptim
    copt::ScalableES.Adam
    popt::ScalableES.Adam

    HrlAdam(cdim::Int, pdim::Int, lr::Real) = new(ScalableES.Adam(cdim, lr), ScalableES.Adam(pdim, lr))
end

optimize(opt::HrlAdam, grad::Vector) = ScalableES.optimize(opt.copt), ScalableES.optimize(opt.popt)

function ScalableES.optimize!(
    π::HrlPolicy,
    ranked::Vector{HrlEsResult{T}},
    nt::NoiseTable,
    optim::HrlAdam,
    l2coeff::Float32,
) where {T<:AbstractFloat}
    cres = map(r -> r.cres, ranked)
    pres = map(r -> r.pres, ranked)
    ScalableES.optimize!(π.cπ, cres, nt, optim.copt, l2coeff), ScalableES.optimize!(π.pπ, pres, nt, optim.popt, l2coeff)
end
