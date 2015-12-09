######### Inference, prediction and optimisation #############
#
#
function doMean(gp::GPmodel,x; wrt=0)
    return doMean(gp.meanfn,x,wrt=wrt)
end

# takes the x vector from the current gp and predictions from the linked gp for
# those ymu values as the values for this mean function
# EXPERIMENTAL
function doMean(meanfn::GPMeanFn,x; wrt=0)
    (ymu,_,_,_,_)=prediction(meanfn.gp,post,x); return ymu;
end

# takes the x vector and rebases (maps) these to the baseint hyper-parameter
# then uses these to get predictions from the linked gp and uses those ymu
# values as the values for this mean function.
# e.g, if x is 10.2365 and baseint is 3, then
# we find the prediction for 3.2365 and use that as the value
# EXPERIMENTAL
function doMean(meanfn::GPMeanUPer,x; wrt=0)
    b = meanfn.baseint
    ox = similar(x)
    for i=1:size(x,1);
        (r,_)=modf(x[i])
        ox[i]=b+r
    end
    (ymu,_,_,_,_)=prediction(meanfn.gp,post,ox)
    return ymu
end

# default mean Fn is MeanZero
function doMean(meanfn::MeanFn,x; wrt=0)
    mn = zeros(eltype(x),size(x,1));
    return mn
end

#l_bounds_dflt=log([0.2;0.2;0.001]);
#u_bounds_dflt=log([3.0;1.8;0.5]);


# These are mainly used to sheppard hypers around for composite kernels
function getHypers(gp::GPmodel)
    covhyps = hypers(gp.covfn)
    lenparms = length(covhyps)+length(gp.likfn.hyp)+length(gp.meanfn.hyp)
    parms = Array(Float64,lenparms)
    i=1
    for j=1:length(covhyps)
        parms[i] = covhyps[j]; i += 1
    end
    for j=1:length(gp.likfn.hyp) # still need to do hypers for these
        parms[i] = gp.likfn.hyp[j]; i += 1
    end
    for j=1:length(gp.meanfn.hyp)
        parms[i] = gp.meanfn.hyp[j]; i += 1
    end
    return parms
end

function setHypers(gp::GPmodel,parms)
    covhyps = hypers(gp.covfn)
    lenparms = length(covhyps)+length(gp.likfn.hyp)+length(gp.meanfn.hyp)
    hypers!(gp.covfn,parms[1:length(covhyps)]); i=length(covhyps)+1
    for j=1:length(gp.likfn.hyp)
        gp.likfn.hyp[j] = parms[i]; i += 1
    end
    for j=1:length(gp.meanfn.hyp)
        gp.meanfn.hyp[j] = parms[i]; i += 1
    end
    return parms
end

# call this do do optimization - presently only using NLopt.jl
function optinf(gp::GPmodel, maxopt; algo=:LD_LBFGS, l_b=Float64[],u_b=Float64[], with_dnlz=true)
    # :LN_COBYLA :LD_LBFGS
    prms = getHypers(gp)
    f = (parms::Vector, grads::Vector) -> optfn(parms,grads,gp,with_dnlz=with_dnlz)
    opt = NLopt.Opt(algo, length(prms));
    #opt = NLopt.Opt(:LN_COBYLA, length(prms));
    NLopt.maxeval!(opt,maxopt)
    if length(l_b) > 0
        NLopt.lower_bounds!(opt,l_b)
    end
    if length(u_b) > 0
        NLopt.upper_bounds!(opt,u_b)
    end
    NLopt.min_objective!(opt, f)
    (optf,optx,ret) = NLopt.optimize(opt,prms)
end

# the objective function - closure over gp which NLopt ignores
function optfn(parms::Vector, grads::Vector, gp::GPmodel; with_dnlz=with_dnlz)
    #println("1------parms-------> ",parms'," w_d:",with_dnlz)
    if !with_dnlz && length(grads) > 0
        error("with_dnlz is false, but len grads > 0")
    end
    setHypers(gp,parms);
    post,nlZ,dnlZ = inference(gp, with_dnlz=with_dnlz)
    if with_dnlz && length(grads) > 0
        for j=1:length(dnlZ)
            grads[j] = dnlZ[j]
        end
    end
    return nlZ[1]
end

# do the inference goodies - the results are returned as a tuple which
# can then be passed on for prediction, i.e returns (post,nlZ,dnlZ)
function inference(gp::GPmodel; with_dnlz=false)
    inference(gp.inffn,gp.meanfn,gp.covfn,gp.likfn,gp.x,gp.y, with_dnlz=with_dnlz)
end

function inference(inffn::InfFn,meanfn::MeanFn,covfn::CovFn,likfn::LikFn,x,y; with_dnlz=false)
    post,nlZ,dnlZ = doInf(inffn::InfFn,meanfn::MeanFn,covfn::CovFn,likfn::LikFn,x,y, with_dnlz=with_dnlz)
    return (post,nlZ,dnlZ)
end

function prediction(gp::GPmodel,post::Post,xs)
    prediction(gp,post,xs,[])
end
function prediction(gp::GPmodel,post::Post,xs,ys)
    prediction(gp.meanfn,gp.covfn,gp.likfn,post,gp.x,xs,ys)
end

# what the future holds...
function prediction(meanfn::MeanFn,covfn::CovFn,likfn::LikFn,post::Post,x,xs)
    return prediction(meanfn,covfn,likfn,post,x,xs,[])
end
function prediction(meanfn::MeanFn,covfn::CovFn,likfn::LikFn,post::Post,x,xs,ys)
    alpha = post.alpha
    L = post.L
    sW = post.sW
    #
    Ks = doCov(covfn,x,xs)
    mns = doMean(meanfn,xs)
    fmu = mns + (alpha'*Ks)'
    #
    kss = doCov(covfn,xs,[]) # diag
    #istril = all((x)->x==0.0,tril!(post.L,-1))
    if istril(post.L)
        v = L'\(sW.*Ks)
        fs2 = kss - sum(v.*v,1)'
    else
        fs2 = kss + sum(Ks.*(L*Ks),1)'
    end
    fs2 = max(fs2,0.0) # remove numerical noise i.e. negative variances
    if length(ys) == 0
        lp,ymu,ys2 = doLik(likfn,[],fmu,fs2)
    else
        lp,ymu,ys2 = doLik(likfn,ys,fmu,fs2)
    end
    return (ymu,ys2,fmu,fs2,lp)
end

# Generate pseudo-random numbers in a quick and dirty way.
# The function makes sure, we obtain the same random numbers using Octave and
# Matlab for the demo scripts.
#
# Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch 2010-07-07.

function gpml_randn(seed::Float64,argin...)
	#nargin = length(argin)
	n = prod(argin); N = int64(ceil(n/2)*2);
	# minimal uniform random number generator for uniform deviates from [0,1]
	# by Park and Miller
	a = 7^5; m = 2^31-1;
	# using Schrage's algorithm
	q = floor(m/a); r = mod(m,a); # m = a*q+r
	u = zeros(N+1,1);
	u[1] = floor(seed*2^31);
	for i=2:N+1
	  # Schrage's algorithm for mod(a*u(i),m)
	  u[i] = a*mod(u[i-1],q) - r*floor(u[i-1]/q);
	  if u[i]<0
	  	u[i] = u[i]+m
	  end
	end
	u = u[2:N+1]/2^31;
	# Box-Muller transform: Numerical Recipies, 2nd Edition, $7.2.8
	# http://en.wikipedia.org/wiki/Box-Muller_transform
	w = sqrt(- 2*log(u[1:N/2]));   # split into two groups
	x = [w.*cos(2*pi*u[N/2+1:N]); w.*sin(2*pi*u[N/2+1:N])];
	x = reshape(x[1:n],argin);
end
