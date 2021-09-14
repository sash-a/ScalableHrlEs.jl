mutable struct HrlPolicy{T} <: ScalableES.AbstractPolicy
    cπ::ScalableES.Policy{T}
    pπ::ScalableES.Policy{T}
end

function HrlPolicy(cnn, pnn, comm::Union{Comm,ScalableES.ThreadComm})
    cp = Policy(cnn, comm)
    pp = Policy(pnn, comm)
    HrlPolicy{Float32}(cp, pp)
end
ScalableES.to_nn(π::HrlPolicy) = ScalableES.to_nn(π.cπ), ScalableES.to_nn(π.pπ)