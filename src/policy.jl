mutable struct HrlPolicy{T} <: ScalableES.AbstractPolicy
    cπ::ScalableES.Policy{T}
    pπ::ScalableES.Policy{T}
end

function HrlPolicy(cnn, pnn)
    cp = Policy(cnn)
    pp = Policy(pnn)
    HrlPolicy{Float32}(cp, pp)
end

ScalableES.to_nn(π::HrlPolicy) = ScalableES.to_nn(π.cπ), ScalableES.to_nn(π.pπ)

function ScalableES.bcast_policy!(π::HrlPolicy, comm::Comm)
    ScalableES.bcast_policy!(π.cπ, comm)
    ScalableES.bcast_policy!(π.pπ, comm)
end
function ScalableES.bcast_policy!(::HrlPolicy, ::ScalableES.ThreadComm) end

# so that it matches the signature of onehot_forward
forward(nn, x, obmean, obstd, cdist, y, s, n, rng) = ScalableES.forward(nn, x, obmean, obstd, rng) * cdist
function onehot_forward(nn, x, obmean, obstd, max_dist, yaw, sensor_span, nbins, rng)
    out = ScalableES.forward(nn, x, obmean, obstd, rng)

    bin_idx = argmax(out)
    dist_percent = out[bin_idx]

    bin_res = sensor_span / nbins
    half_span = sensor_span / 2f0
    n_bins_inv = 1f0 / nbins

    angle = bin_idx * bin_res - half_span + n_bins_inv + Float32(yaw)
    return pol2cart(max(dist_percent * max_dist, 1), angle)
end