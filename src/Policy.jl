mutable struct HrlPolicy <: ScalableES.AbstractPolicy
    cπ::ScalableES.Policy
    pπ::ScalableES.Policy
end

HrlPolicy(cnn::Chain, pnn::Chain) = HrlPolicy(Policy(Flux.destructure(cnn)...), Policy(Flux.destructure(pnn)...))
ScalableES.to_nn(π::HrlPolicy) = ScalableES.to_nn(π.cπ), ScalableES.to_nn(π.pπ)