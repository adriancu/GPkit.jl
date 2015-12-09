# GPML - liKFunctions - prediction mode
# With three or four input arguments:                       [PREDICTION MODE]
#
#    lp = lik(hyp, y, mu) OR [lp, ymu, ys2] = lik(hyp, y, mu, s2)
#
# This allows to evaluate the predictive distribution. Let p(y_*|f_*) be the
# likelihood of a test point and N(f_*|mu,s2) an approximation to the posterior
# marginal p(f_*|x_*,x,y) as returned by an inference method. The predictive
# distribution p(y_*|x_*,x,y) is approximated by.
#   q(y_*) = \int N(f_*|mu,s2) p(y_*|f_*) df_*
#
#   lp = log( q(y) ) for a particular value of y, if s2 is [] or 0, this
#                    corresponds to log( p(y|mu) )
#   ymu and ys2      the mean and variance of the predictive marginal q(y)
#                    note that these two numbers do not depend on a particular 
#                    value of y 
#  All vectors have the same size.
#

# GPML - liKFunctions - inference mode
# With five or six input arguments, the fifth being a string [INFERENCE MODE]
#
# [varargout] = lik(hyp, y, mu, s2, inf) OR
# [varargout] = lik(hyp, y, mu, s2, inf, i)
#
# There are three cases for inf, namely a) infLaplace, b) infEP and c) infVB. 
# The last input i, refers to derivatives w.r.t. the ith hyperparameter. 
#
# a1) [lp,dlp,d2lp,d3lp] = lik(hyp, y, f, [], 'infLaplace')
# lp, dlp, d2lp and d3lp correspond to derivatives of the log likelihood 
# log(p(y|f)) w.r.t. to the latent location f.
#   lp = log( p(y|f) )
#  dlp = d   log( p(y|f) ) / df
# d2lp = d^2 log( p(y|f) ) / df^2
# d3lp = d^3 log( p(y|f) ) / df^3
#
# a2) [lp_dhyp,dlp_dhyp,d2lp_dhyp] = lik(hyp, y, f, [], 'infLaplace', i)
# returns derivatives w.r.t. to the ith hyperparameter
#   lp_dhyp = d   log( p(y|f) ) / (     dhyp_i)
#  dlp_dhyp = d^2 log( p(y|f) ) / (df   dhyp_i)
# d2lp_dhyp = d^3 log( p(y|f) ) / (df^2 dhyp_i)
#
#
# b1) [lZ,dlZ,d2lZ] = lik(hyp, y, mu, s2, 'infEP')
# let Z = \int p(y|f) N(f|mu,s2) df then
#   lZ =     log(Z)
#  dlZ = d   log(Z) / dmu
# d2lZ = d^2 log(Z) / dmu^2
#
# b2) [dlZhyp] = lik(hyp, y, mu, s2, 'infEP', i)
# returns derivatives w.r.t. to the ith hyperparameter
# dlZhyp = d log(Z) / dhyp_i
#


function doLik(likfn::LikGauss,y,mu)
    sn2 = exp(2.0*likfn.hyp[1])
    y = length(y)==0 ? fill!(similar(mu),0.0) : y
    lp = -(y-mu).^2.0./sn2/2.0-log(2.0*pi*sn2)/2.0
    ymu = mu            # first y moment
    ys2 = sn2           # second y moment
    return lp,ymu,ys2
end

function doLik(likfn::LikGauss,y,mu,s2)
    if norm(s2) <= 0.0
        return doLik(likfn,y,mu)
    end
    sn2 = exp(2.0*likfn.hyp[1])
    y = length(y)==0 ? fill!(similar(mu),0.0) : y
    lp = -(y-mu).^2.0./(sn2.+s2)/2.0 - log(2.0*pi*(sn2.+s2))/2.0  # log partition function
    ymu = mu
    ys2 = s2 .+ sn2
    return lp,ymu,ys2
end

function doLik(likfn::LikGauss,y,mu,s2,inffn; wrt=0)
    sn2 = exp(2*likfn.hyp[1])
    y = length(y)==0 ? fill!(similar(mu),0.0) : y
    if typeof(inffn) == InfEP  # inf is typeof(inffn)
        if wrt == 0    # no derivative mode
            lZ = -(y-mu).^2.0./(sn2+s2)/2.0 - log(2.0*pi*(sn2+s2))/2.0  # log partition function
            dlZ  = (y-mu)./(sn2+s2)     # 1st derivative w.r.t. mean
            d2lZ = -1.0./(sn2+s2)       # 2nd derivative w.r.t. mean
    #println("-lZ->",lZ)
            return lZ,dlZ,d2lZ
        else   # derivative mode
            dlZhyp = ((y-mu).^2.0./(sn2+s2)-1.0) ./ (1.0+s2./sn2)   # deriv. w.r.t. likfn.hyp
    #println("-dlZhyp->",dlZhyp)
            return dlZhyp
        end
#    elseif typeof(inffn) == InfLaplace
#        if wrt == 0    # no derivative mode
#            ymmu = y-mu
#            lp = -ymmu.^2.0/(2.0*sn2) - log(2.0*pi*sn2)/2.0 
#            dlp = ymmu/sn2      # dlp, derivative of log likelihood
#            d2lp = -fill!(similar(ymmu),1.0)/sn2
#            d3lp = fill!(similar(ymmu),0.0)
#            return lp,dlp,d2lp,d3lp
#        else   # derivative mode
#           lp_dhyp = (y-mu).^2.0/sn2 - 1.0  # derivative of log likelihood w.r.t. hyp
#           dlp_dhyp = 2.0*(mu-y)/sn2        # first derivative
#           d2lp_dhyp = 2.0*fill!(similar(ymmu),1.0)/sn2   # and of second mu derivative
#            return lp_dhyp,dlp_dhyp,d2lp_dhyp
#        end
    else
        error("No LikGauss support for inference: "*string(inf))
    end
end

