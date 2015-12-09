
function doInf(inffn::InfExact,meanfn::MeanFn,covfn::CovFn,likfn::LikFn,x,y; with_dnlz=false)
    if typeof(likfn) != LikGauss; error("InfExact only possible with likGauss"); end
    sn2 = exp(likfn.hyp[1]*2.0)
    sn2 = sn2 == 0.0 ? eps()*10e3 : sn2 # is this correct ???
    n = size(x,1)
    K = doCov(covfn,x,x)
    L = chol(K/sn2+eye(n),:U)
    mn = doMean(meanfn,x)
    ymdif = y-mn
    #alpha = L'\(L\ymdif)/sn2 - if we'd used :L above
	alpha = L\(L'\ymdif)/sn2
	sW = ones(n,1)/sqrt(sn2)    # sqrt of the noise precision vector (as per gpml)
    # neg log marg likelihood: nlZ
    nlZ = ymdif'*alpha/2.0 .+ sum(log(diag(L))) .+ n*log(2*pi*sn2)/2.0
    if with_dnlz
        covhyps = hypers(covfn)
        dnlz = Array(eltype(covhyps),length(covhyps)+length(likfn.hyp)+length(meanfn.hyp)) # allocate array
        #show(dnlz)
        Q = L\(L'\eye(n))/sn2 - alpha*alpha'
        ii = 0
        for i = 1:length(covhyps)
            ii+=1
            dnlz[ii] = sum(sum(Q.*doCov(covfn,x,x,wrt=i)))/2.0
        end
        for i = 1:1 # length(likfn.hyp) -> always 1
            ii+=1
            dnlz[ii] = sn2*trace(Q)
        end
        for i = 1:length(meanfn.hyp)
            ii+=1
            dnlz[ii] = -doMean(meanfn,x,wrt=i)'*alpha
        end
    else
        dnlz = Float64[]
    end
    post = Post(alpha, sW, L)
    return post, nlZ, dnlz
end