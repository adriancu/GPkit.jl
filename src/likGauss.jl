
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
