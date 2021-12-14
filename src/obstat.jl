import Base: +

struct HrlObstat{CS,PS,T} <: ScalableES.AbstractObstat
    cobstat::ScalableES.Obstat{CS,T}
    pobstat::ScalableES.Obstat{PS,T}
end

HrlObstat(cshape, pshape, eps::Float32) =
    HrlObstat{cshape,pshape,Float32}(ScalableES.Obstat(cshape, eps), ScalableES.Obstat(pshape, eps))

# ScalableES.make_obstat(shape, ::HrlPolicy) = HrlObstat(shape, shape + 3, 0f0)
function ScalableES.make_obstat(shape, pol::HrlPolicy)
    (cnn, pnn) = ScalableES.to_nn(pol)  # this is kinda expensive, but only called once per generation
    HrlObstat(shape, last(size(first(pnn.layers).W)), 0.0f0)
end

function ScalableES.add_obs(obstat::HrlObstat{CS,PS,T}, obs) where {CS} where {PS} where {T<:AbstractFloat}
    HrlObstat(ScalableES.add_obs(obstat.cobstat, first(obs)), ScalableES.add_obs(obstat.pobstat, last(obs)))
end

+(x::HrlObstat{CS,PS,T}, y::HrlObstat{CS,PS,T}) where {CS} where {PS} where {T<:AbstractFloat} =
    HrlObstat{CS,PS,T}(x.cobstat + y.cobstat, x.pobstat + y.pobstat)
ScalableES.mean(o::HrlObstat{CS,PS,T}) where {CS} where {PS} where {T<:AbstractFloat} = mean(o.cobstat), mean(o.pobstat)
ScalableES.std(o::HrlObstat{CS,PS,T}) where {CS} where {PS} where {T<:AbstractFloat} = std(o.cobstat), std(o.pobstat)
