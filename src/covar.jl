
function sq_dist(a,b)
    (D,n) = size(a)
    (d,m) = size(b)
    mu = (m/(n+m))*mean(b,2) + (n/(n+m))*mean(a,2)
    a = a.-mu
    b = b.-mu
    C = sum(a.*a,1)' .+ sum(b.*b,1).-(2*a'*b)
    return C
end
sq_dist(a)=sq_dist(a,a)


function doCov(gp::GPmodel,a,b; wrt=0)
    doCov(gp.covfn,a,b,wrt=wrt)
end

function doCov(cov::CovSEiso,a,b; wrt=0)
    cls = exp(cov.hyp[1]);      # characteristic length scale
    sf2 = exp(2*cov.hyp[2]);    # signal variance
    b = size(b)==()?[b]:b
    if length(b) == 0
        K = zeros(size(a,1),1)
    else
        K = sq_dist(a'/cls,b'/cls)
    end
    if wrt == 0
        K = sf2*exp(-K/2)
    elseif wrt == 1
        K = sf2*exp(-K/2).*K
    else
        K = 2*sf2*exp(-K/2)
    end
    return K
end

function doCov(cov::CovSEard,a,b; wrt=0)
    n,D = size(a)
    ell = exp(cov.hyp[1:D]) # characteristic length scale
    sf2 = exp(2*cov.hyp[D+1])

    b = size(b)==()?[b]:b
    if length(b) == 0
        K = zeros(size(a,1),1)
    else
        K = sq_dist(diagm(1./ell)*a',diagm(1./ell)*b')
    end
    K = sf2*exp(-K/2)  # covariance
    if wrt > 0
        if wrt <= D
            if length(b) == 0 # diagonal
                K = K*0
            else
                if a == b
                    K = K.*sq_dist(a[:,wrt]'/ell[wrt])
                else
                    K = K.*sq_dist(a[:,wrt]'/ell[wrt],b[:,i]'/ell[wrt])
                end
            end
        elseif wrt == D+1
            K = 2*K
        else
            error("Unknown hyperparameter")
        end
    end
    return K
end

function doCov(cov::CovMaterniso,a,b; wrt=0)
    cls = exp(cov.hyp[1])       # characteristic length scale
    sf2 = exp(2.0*cov.hyp[2])   # signal variance
    b = size(b)==()?[b]:b
    d = cov.v
    if d == 1
        f = (t)-> 1.0
        df = (t)-> 1.0
    elseif d == 3
        f = (t)-> 1.0.+t
        df = (t)-> t
    elseif d == 5
        f = (t)-> 1.0.+t.*(1.0.+t./3.0)
        df = (t)->t.*(1.0.+t)./3.0
    else
        error("cov.v must be 1,3 or 5")
    end
    m = (t,f)-> f(t).*exp(-t)
    dm = (t,f)-> df(t).*t.*exp(-t)

    if length(b) == 0
        K = zeros(size(a,1),1)
    else
        K = sqrt(sq_dist(sqrt(d).*a'./cls,sqrt(d)*b'./cls))
    end
    if wrt == 0
        K = sf2.*m(K,f)
    elseif wrt == 1
        K = sf2.*dm(K,f)
    elseif wrt == 2
        K = 2*sf2.*m(K,f)
    else
        error("Invalid derivative wrt request")
    end
    return K
end

function doCov(cov::CovPeriodic,a,b; wrt=0)
    cls = exp(cov.hyp[1]);
    p   = exp(cov.hyp[2]);
    sf2 = exp(2*cov.hyp[3]);

    b = size(b)==()?[b]:b
    if length(b) == 0
        K = zeros(size(a,1),1)
    else
        K = sqrt(sq_dist(a',b'))
    end

    K = pi*K/p
#show(size(K))
    if wrt == 0
        K = sin(K)/cls; K = K.*K; K = sf2*exp(-2*K)
    elseif wrt == 1 # drv wrt hyp1
        K = sin(K)/cls; K = K.*K; K = 4*sf2*exp(-2*K).*K
    elseif wrt == 2 # ditto hyp2
        R = sin(K)/cls; K = 4*sf2/cls*exp(-2*R.*R).*R.*cos(K).*K;
    elseif wrt == 3 # ditto hyp3
        K = sin(K)/cls; K = K.*K; K = 2*sf2*exp(-2*K);
    else
        error("Invalid derivative wrt request")
    end
end

function doCov(cov::CovRQiso,a,b; wrt=0)
    cls = exp(cov.hyp[1]);      # characteristic length scale
    sf2 = exp(2*cov.hyp[2]);    # signal variance
    alpha = exp(2*cov.hyp[3]);    # signal variance
    b = size(b)==()?[b]:b
    if length(b) == 0
        D2 = zeros(size(a,1),1)
    else
        D2 = sq_dist(a'/cls,b'/cls)
    end
    if wrt == 0
        K = sf2*((1+0.5*D2/alpha).^(-alpha))
    elseif wrt == 1
        K = sf2*(1+0.5*D2/alpha).^(-alpha-1).*D2
    elseif wrt == 2
        K = 2*sf2*((1+0.5*D2/alpha).^(-alpha));
    else
        K = (1+0.5*D2/alpha);
        K = sf2*K.^(-alpha).*(0.5*D2./K - alpha*log(K));
    end
    return K
end


function doCov(cov::CovSum,a,b; wrt=0)
    #println("wrt=$wrt")
    if wrt == 0
        i = 1
        K = doCov(cov.covs[i],a,b,wrt=wrt)
        i += 1
        while i <= length(cov.covs)
            K = K + doCov(cov.covs[i],a,b,wrt=wrt)
            i += 1
        end
        return K
    else
        i = 1
        for c = 1:length(cov.covs)
            hyps = hypers(cov.covs[c])
            for h = 1:length(hyps)
                if i == wrt
                    K = doCov(cov.covs[c],a,b,wrt=h)
                    return K
                end
                i += 1
            end
        end
    end
    error("Invalid wrt in covsum")
end

#TBD - gmpl - below derivatives is just a copy of sum - prod is more complex
function doCov(cov::CovProd,a,b; wrt=0)
    #println("wrt=$wrt")
    if wrt == 0
        i = 1
        K = doCov(cov.covs[i],a,b,wrt=wrt)
        i += 1
        while i <= length(cov.covs)
            K = K .* doCov(cov.covs[i],a,b,wrt=wrt)
            i += 1
        end
        return K
    else
        error("no derivatives for now.")
#        i = 1
#        for c = 1:length(cov.covs)
#            hyps = hypers(cov.covs[c])
#            for h = 1:length(hyps)
#                if i == wrt
#                    K = doCov(cov.covs[c],a,b,wrt=h)
#                    return K
#                end
#                i += 1
#            end
#        end
    end
    error("Invalid wrt in covsum")
end

function doCov(cov::CovLIN,a,b; wrt=0)
    # k(x^p,x^q) = x^p'*x^q
    if length(b) == 0
        K = sum(a.*a,2)
    else
        K = a*b'
    end
    return K
end

function hypers(cov::CovSum)
    hyp = Float64[]
    for i=1:length(cov.covs)
        h = hypers(cov.covs[i])
        for j=1:length(h)
            push!(hyp,h[j])
        end
    end
    return hyp
end
function hypers!(cov::CovSum,hyp::AbstractArray)
    k = 1
    for i=1:length(cov.covs)
        hy = hypers(cov.covs[i])
        hypers!(cov.covs[i],hyp[k:k+length(hy)-1])
        k += length(hy)
    end
    return k-1
end
function hypers(cov::CovProd)
    hyp = Float64[]
    for i=1:length(cov.covs)
        h = hypers(cov.covs[i])
        for j=1:length(h)
            push!(hyp,h[j])
        end
    end
    return hyp
end
function hypers!(cov::CovProd,hyp::AbstractArray)
    k = 1
    for i=1:length(cov.covs)
        hy = hypers(cov.covs[i])
        hypers!(cov.covs[i],hyp[k:k+length(hy)-1])
        k += length(hy)
    end
    return k-1
end
function hypers(cov::CovFn)
    return cov.hyp
end
function hypers!(cov::CovFn,hyp::AbstractArray)
    cov.hyp = hyp
end
