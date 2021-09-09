mutable struct HrlAdam <: ScalableES.AbstractOptim
    copt::ScalableES.Adam
    popt::ScalableES.Adam

    HrlAdam(cdim::Int, pdim::Int, lr::Real) = new(ScalableES.Adam(cdim, lr), ScalableES.Adam(pdim, lr))
end

optimize(opt::HrlAdam, grad::Vector) = ScalableES.optimize(opt.copt), ScalableES.optimize(opt.popt)