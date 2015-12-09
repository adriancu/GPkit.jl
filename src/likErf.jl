
# Problems with conversion of the supporting functions below - i.e conversion
#of lp(ok) to lp[ok], etc

#function [varargout] = likErf(hyp, y, mu, s2, inf, i)

# likErf - Error function or cumulative Gaussian likelihood function for binary
# classification or probit regression. The expression for the likelihood is 
#   likErf(t) = (1+erf(t/sqrt(2)))/2 = normcdf(t).
#
# Several modes are provided, for computing likelihoods, derivatives and moments
# respectively, see likFunctions.m for the details. In general, care is taken
# to avoid numerical issues when the arguments are extreme.
# 
# Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-07-22.
#

function doLik(likfn::LikErf,y,mu)
    lp,ymu,ys2 = doLik(likfn,y,mu,0.0)
    return lp,ymu,ys2
end

function doLik(likfn::LikErf,y,mu,s2)
    y = sign(y)
    for i=1:length(y)
        y[i] = y[i]==0 ? 1.0 : y[i]
    end
    if norm(s2) <= 0.0
        p,lp = cumGauss(y,mu)
    else
        z = mu./sqrt(1+s2)
        junk,lZ = cumGauss(y,z)                             # log part function
        lp = lZ; p = exp(lp)
    end
    ymu = 2*p-1
    ys2 = 4*p.*(1-p)
    return lp,ymu,ys2
end

function doLik(likfn::LikErf,y,mu,s2,inffn; wrt=0)

    y = length(y)==0 ? fill!(similar(mu),0.0) : y
    if typeof(inffn) == InfEP # inf is typeof(inffn)
        if wrt == 0    # no derivative mode
            z = mu./sqrt(1+s2)
            junk,lZ = cumGauss(y,z)                             # log part function
            yy = length(y)==0 ? 1.0 : y
            n_p = gauOverCumGauss(z,exp(lZ))
            dlZ = yy.*n_p./sqrt(1.0+s2)                  # 1st derivative wrt mean
            d2lZ = -n_p.*(z+n_p)./(1.0+s2)               # 2nd derivative wrt mean
            return lZ,dlZ,d2lZ
        else   # derivative mode
            dlZhyp = []   # deriv. w.r.t. likfn.hyp
            return dlZhyp
        end
#    elseif typeof(inffn) == InfLaplace
#        if wrt == 0    # no derivative mode
#            f = mu; yf = y.*f               # product latents and labels
#            p,lp = cumGauss(y,f)
#            n_p = gauOverCumGauss(yf,p)
#            dlp = y.*n_p
#            d2lp = -n_p.^2 - yf.*n_p
#            d3lp = 2*y.*n_p.^3 +3*f.*n_p.^2 +y.*(f.^2-1).*n_p
#            return lp,dlp,d2lp,d3lp
#        else   # derivative mode
#            return [],[],[]
#        end
    else
        error("No LikErf support for inference: "*string(inf))
    end
end


function cumGauss(y,f)
    yf = length(y)>0 ? y.*f : f         # product of latents and labels
    p  = (1+erf(yf/sqrt(2.0)))/2.0      # likelihood
    lp = logphi(yf,p)                   # log likelihood
    return p,lp
end

# safe implementation of the log of phi(x) = \int_{-\infty}^x N(f|0,1) df
# logphi(z) = log(normcdf(z))
function logphi(z,p)
    lp = zeros(Float64,size(z))
    zmin = -6.2; zmax = -5.5
    ok = z.>zmax                     # safe evaluation for large values
    bd = z.<zmin                     # use asymptotics
    ip = !ok & !bd;                 # interpolate between both of them
    lam = 1./(1+exp( 25*(1/2-(z[ip]-zmin)/(zmax-zmin)) ));   # interp. weights
    lp[ok] = log( p[ok] );
    # use lower and upper bound acoording to Abramowitz&Stegun 7.1.13 for z<0
    # lower -log(pi)/2 -z.^2/2 -log( sqrt(z.^2/2+2   ) -z/sqrt(2) )
    # upper -log(pi)/2 -z.^2/2 -log( sqrt(z.^2/2+4/pi) -z/sqrt(2) )
    # the lower bound captures the asymptotics
    lp[!ok] = -log(pi)/2 -z[!ok].^2/2 -log( sqrt(z[!ok].^2/2+2)-z[!ok]/sqrt(2) );
    lp[ip] = (1-lam).*lp[ip] + lam.*log( p[ip] );
    return lp
end

function gauOverCumGauss(f,p)
    n_p = zeros(Float64,size(f));       # safely compute Gaussian over cumulative Gaussian
    ok = f.>-5;                  # naive evaluation for large values of f
    n_p[ok] = (exp(-f[ok].^2/2)/sqrt(2*pi)) ./ p[ok]; 

    bd = f.<-6;                   # tight upper bound evaluation
    n_p[bd] = sqrt(f[bd].^2/4+1)-f[bd]/2;

    interp = !ok & !bd;           # linearly interpolate between both of them
    tmp = f[interp];
    lam = -5-f[interp];
    n_p[interp] = [1-lam].*(exp(-tmp.^2/2)/sqrt(2*pi))./p[interp] + 
                lam .*(sqrt(tmp.^2/4+1)-tmp/2);
    return n_p
end


