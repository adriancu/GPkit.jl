module GPkit

export
  CovFn, MeanFn, LikFn, InfFn,
  GPmodel,
  Post,
  GPMeanFn, GPMeanUPer,
  InfExact, infEP, InfLaplace,
  LikGauss, LikLogistic, LikErf,
  MeanZero,
  CovSEiso, CovMaterniso, CovRQiso, CovLIN, CovSEard, CovPeriodic,
  CovSum, CovProd,
  doMean, doCov, doInf, inference, prediction, optinf,
  getHypers, setHypers,
  gpml_randn # can get identical randoms from julia and matlab

const err_numhyps="Wrong number of hyper-parameters"

abstract CovFn
abstract MeanFn
abstract LikFn
abstract InfFn

type GPmodel
    inffn::InfFn
    covfn::CovFn
    likfn::LikFn
    meanfn::MeanFn
    x
    y
end
GPmodel() = GPmodel(InfExact(),CovSEiso(1.0,1.0),LikGauss(0.05),MeanZero(),Float64[],Float64[])

type Post
    alpha
    sW
    L
    Post(alpha,sW,L) = new(alpha,sW,L)
end

type GPMeanFn <: MeanFn
    gp
    post
    GPMeanFn(gp::GPmodel,post::Post) = new(gp,post)
end
type GPMeanUPer <: MeanFn
    gp
    post
    baseint
    GPMeanUPer(gp::GPmodel,post::Post) = new(gp,post)
end

type InfExact <: InfFn
    InfExact() = new()
end

# experimental
type InfEP <: InfFn
    last_ttau::AbstractArray
    last_tnu::AbstractArray
    InfEP() = new(Float64[],Float64[])
end

# incomplete
type InfLaplace <: InfFn
    last_alpha::AbstractArray
    InfLaplace() = new(Float64[])
end

type LikGauss <: LikFn
    hyp::AbstractArray  # sn2
    LikGauss(hyp) = length(hyp) != 1 ? error(err_numhyps) : new(log(hyp))
    LikGauss(hyp::Real) = new(log([hyp]))
end

# not yet
type LikLogistic <: LikFn
    hyp::AbstractArray  # sn2
    LikLogistic(hyp) = length(hyp) != 1 ? error(err_numhyps) : new(log(hyp))
    LikLogistic(hyp::Real) = new(log([hyp]))
end

# not yet
type LikErf <: LikFn
    hyp::AbstractArray  # sn2
    inf::Symbol
    LikErf() = new(Float64[],:nil)
    LikErf(infsym) = new(Float64[],infsym)
end

type MeanZero <: MeanFn
    hyp::AbstractArray
    MeanZero() = new(Float64[])
end

type CovSEiso <: CovFn
    hyp::AbstractArray
    CovSEiso(hyp) = length(hyp) != 2 ? error(err_numhyps) : new(log(hyp))
    CovSEiso(hyp...) = length(hyp) != 2 ? error(err_numhyps) : new(log([hyp[1],hyp[2]]))
end
type CovSEard <: CovFn
    hyp::AbstractArray
    CovSEard(hyp) = length(hyp) < 2 ? error(err_numhyps) : new(log(hyp))
end

type CovMaterniso <: CovFn
    hyp::AbstractArray
    v
    CovMaterniso(hyp;v=3) = length(hyp) != 2 ? error(err_numhyps) : new(log(hyp),v)
    CovMaterniso(hyp...;v=3) = length(hyp) != 2 ? error(err_numhyps) : new(log([hyp[1],hyp[2]]),v)
end

type CovPeriodic <: CovFn
    hyp::AbstractArray
    CovPeriodic(hyp) = length(hyp) != 3 ? error(err_numhyps) : new(log(hyp))
    CovPeriodic(hyp...) = length(hyp) != 3 ? error(err_numhyps) : new(log([hyp[1],hyp[2],hyp[3]]))
end

type CovRQiso <: CovFn
    hyp::AbstractArray
    CovRQiso(hyp) = length(hyp) != 3 ? error(err_numhyps) : new(log(hyp))
    CovRQiso(hyp...) = length(hyp) != 3 ? error(err_numhyps) : new(log([hyp[1],hyp[2],hyp[3]]))
end

type CovLIN <: CovFn
    hyp::AbstractArray
    CovLIN(hyp) = length(hyp) != 0 ? error(err_numhyps) : new(Float64[])
    CovLIN() = new(Float64[])
end

type CovSum <: CovFn
    covs::Array{CovFn,1}
    CovSum(covs) = length(covs) > 0 ? new(covs) : error("Must have > 0 covfns to sum")
end

type CovProd <: CovFn
    covs::Array{CovFn,1}
    CovProd(covs) = length(covs) > 0 ? new(covs) : error("Must have > 0 covfns for product of covs")
end

include("covar.jl")
include("infExact.jl")
#include("infEP.jl")
include("LikGauss.jl")
#include("LikErf.jl")
include("gp_impl.jl")
using NLopt

end # of module
