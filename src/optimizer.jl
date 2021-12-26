mutable struct HrlAdam <: ScalableES.AbstractOptim
    copt::ScalableES.Adam
    popt::ScalableES.Adam

    HrlAdam(cdim::Int, pdim::Int, lr::Real) = new(ScalableES.Adam(cdim, lr), ScalableES.Adam(pdim, lr))
end

function ScalableES.optimize!(
    π::HrlPolicy,
    ranked::Vector{HrlEsResult{T}},
    nt::NoiseTable,
    optim::HrlAdam,
    l2coeff::Float32,
) where {T<:AbstractFloat}
    # ScalableES.optimize!(π.cπ, map(r->r.cres, ranked), nt, optim.copt, l2coeff), ScalableES.optimize!(π.pπ, map(r->r.pres, ranked), nt, optim.popt, l2coeff)
    @show length(π.cπ.θ) length(π.pπ.θ)
    cgrad = l2coeff * π.cπ.θ - ScalableES.approxgrad(π.cπ, nt, map(r -> r.cres, ranked)) ./ (length(ranked) * 4)
    π.cπ.θ .+= ScalableES.optimize(optim.copt, cgrad)

    # pgrad = l2coeff * π.pπ.θ - ScalableES.approxgrad(π.pπ, nt, map(r -> r.pres, ranked)) ./ (length(ranked) * 4)
    # π.pπ.θ .+= ScalableES.optimize(optim.popt, pgrad)
end
