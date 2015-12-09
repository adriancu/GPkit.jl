######## Experimental - incomplete #############
#
# comments are from GPML...
# Expectation Propagation approximation to the posterior Gaussian Process.
# The function takes a specified covariance function (see covFunction.m) and
# likelihood function (see likFunction.m), and is designed to be used with
# gp.m. See also infFunctions.m. In the EP algorithm, the sites are
# updated in random order, for better performance when cases are ordered
# according to the targets.
#
# Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch 2012-11-07.
# Adapted for Julia - Adrian Cuthbertson - 2014-01-01
#
function doInf(inffn::InfEP,meanfn::MeanFn,covfn::CovFn,likfn::LikFn,x,y; with_dnlz=false)
    tol = 1e-4; max_sweep = 10; min_sweep = 2     # tolerance to stop EP iterations
    smax = 2; Nline = 10; thr = 1e-4   # line search parameters
    maxit = 20       # max number of Newton steps in f
    n = size(x,1)
    K = doCov(covfn,x,x)     # evaluate the covariance matrix
    m = doMean(meanfn, x)    # evaluate the mean vector

    # A note on naming: variables are given short but descriptive names in
    # accordance with Rasmussen & Williams "GPs for Machine Learning" (2006): mu
    # and s2 are mean and variance, nu and tau are natural parameters. A leading t
    # means tilde, a subscript _ni means "not i" (for cavity parameters), or _n
    # for a vector of cavity parameters.

    # marginal likelihood for ttau = tnu = zeros(n,1); equals n*log(2) for likCum*
    #println("foo-->")
    #println(likfn.hyp)
    nlZ0 = -sum( doLik(likfn,y,m,diag(K),inffn)[1] )
    #println("nlZ0->",nlZ0)
    if length(inffn.last_ttau) == 0
        ttau = fill!(similar(x),0.0)
        tnu = fill!(similar(x),0.0)
        Sigma = K                    #initialize Sigma and mu, the parameters of
        mu = fill!(similar(x),0.0)   # .. the Gaussian posterior approximation
        nlZ = nlZ0
    else
        ttau = inffn.last_ttau               # try the tilde values from previous call
        tnu  = inffn.last_tnu
        Sigma,mu,nlZ,L = epComputeParams(K,y,ttau,tnu,likfn,m)
        if nlZ[1] > nlZ0[1]                   # if zero is better ..
            ttau = fill!(similar(x),0.0)    # .. then initialize with zero instead
            tnu  = fill!(similar(x),0.0)
            Sigma = K                    # initialize Sigma and mu, the parameters of ..
            mu = fill!(similar(x),0.0)   # .. the Gaussian posterior approximation
            nlZ = nlZ0
        end
    end


    nlZ_old = Inf; sweep = 0         # converged, max. sweeps or min. sweeps?

    cnd0 = abs(nlZ[1]-nlZ_old[1]) > tol
    while (cnd0 && sweep < max_sweep) || sweep < min_sweep
        nlZ_old = nlZ; sweep = sweep+1
        for i in randperm(n)       # iterate EP updates (in random order) over examples
            tau_ni = 1/Sigma[i,i]-ttau[i];      #  first find the cavity distribution ..
            nu_ni = mu[i]/Sigma[i,i]+m[i]*tau_ni-tnu[i];    # .. params tau_ni and nu_ni

            # compute the desired derivatives of the indivdual log partition function
            lZ,dlZ,d2lZ = doLik(likfn,y[i],nu_ni/tau_ni,1.0/tau_ni,inffn)
            #println("lZ,dlZ,d2lZ->",i," ",lZ," ",dlZ," ",d2lZ)
            ttau_old = ttau[i];   # then find the new tilde parameters, keep copy of old

            ttau[i] = -d2lZ / (1.0+d2lZ/tau_ni)
            ttau[i] = max(ttau[i],0.0); # enforce positivity i.e. lower bound ttau by zero
            tnu[i]  = ( dlZ + (m[i]-nu_ni/tau_ni)*d2lZ )/(1.0+d2lZ/tau_ni)

            ds2 = ttau[i] - ttau_old                    # finally rank-1 update Sigma ..
            si = Sigma[:,i]
            Sigma = Sigma - ds2/(1.0+ds2*si[i])*si*si'  # takes 70% of total time
            mu = Sigma*tnu                              # .. and recompute mu
        end
        # recompute since repeated rank-one updates can destroy numerical precision
        Sigma,mu,nlZ,L = epComputeParams(K,y,ttau,tnu,likfn,m)

        #println(typeof(nlZ)," ",typeof(nlZ_old)," ",tol)
        cnd0 = abs(nlZ[1]-nlZ_old[1]) > tol
    end

    if sweep == max_sweep
        error("maximum number of sweeps reached in function infEP")
    end

    inffn.last_ttau = ttau; inffn.last_tnu = tnu            #remember for next call

    # solve_chol: X = L\(L'\B)
    sW = sqrt(ttau);
    bbb = sW.*(K*tnu)
    ccc = L\(L'\bbb)
    alpha = tnu-sW.*ccc   #solve_chol(L,sW.*(K*tnu))
    post = Post(alpha, sW, L)          # return the posterior params
    if with_dnlz                       # do we want derivatives?
        #println("covfn.hyp->")
        #println(covfn.hyp)
        dnlz = Array(eltype(covfn.hyp),length(covfn.hyp)+length(likfn.hyp)+length(meanfn.hyp)) # allocate array
        V = L'\(repmat(sW,1,n).*K)
        Sigma = K - V'*V
        mu = Sigma*tnu
        tau_n = 1.0/diag(Sigma)-ttau            # compute the log marginal likelihood
        nu_n  = mu./diag(Sigma)-tnu+m.*tau_n    # vectors of cavity parameters
        # solve_chol: X = L\(L'\B)
        F = alpha*alpha'-repmat(sW,1,n).* (L\(L'\diagm(sW)))  #solve_chol(L,diag(sW));
        # covariance hypers
        ii = 0
        for i=1:length(covfn.hyp)
            dK = doCov(covfn,x,x,wrt=i)
            ii+=1
            dnlz[ii] = -sum(sum(F.*dK))/2.0;
            #println("F.*dK: ",F.*dK)
            #println("sum(F.*dK): ",sum(F.*dK))
            #println("dnlz[ii]: ",dnlz[ii])
        end
        # likelihood hypers
        for i = 1:length(likfn.hyp)
            dlik = doLik(likfn, y, nu_n./tau_n, 1.0./tau_n,inffn, wrt=i)
            ii+=1
            dnlz[ii] = -sum(dlik)
        end
        junk,dlZ = doLik(likfn, y, nu_n./tau_n, 1./tau_n,inffn) # mean hyps
        for i = 1:length(meanfn.hyp)
            dm = doMmean(meanfn, x, wrt=i)
            ii+=1
            dnlz[ii] = -dlZ'*dm;
        end
    else
        dnlz = Float64[]
    end
    #println("nlZ->",nlZ)
    #println("dnlz->",dnlz)
    return post,nlZ,dnlz
end

# function to compute the parameters of the Gaussian approximation, Sigma and
# mu, and the negative log marginal likelihood, nlZ, from the current site
# parameters, ttau and tnu. Also returns L (useful for predictions).
function epComputeParams(K,y,ttau,tnu,likfn,m)
    n = length(y)                    # number of training cases
    sW = sqrt(ttau)                  # compute Sigma and mu
    L = chol(eye(n)+sW*sW'.*K,:U)       # L'*L=B=eye(n)+sW*K*sW
    V = L'\(repmat(sW,1,n).*K)
    Sigma = K - V'*V
    mu = Sigma*tnu
    tau_n = 1./diag(Sigma)-ttau      # compute the log marginal likelihood
    nu_n  = mu./diag(Sigma)-tnu+m.*tau_n    # vectors of cavity parameters

    lZ = doLik(likfn, y, nu_n./tau_n, 1./tau_n,inffn)[1]
#    nlZ = sum(log(diag(L))) -sum(lZ) -tnu'*Sigma*tnu/2.0
#        -(nu_n-m.*tau_n)'*((ttau./tau_n.*(nu_n-m.*tau_n)-2.0*tnu)./(ttau+tau_n))/2.0
#        +sum(tnu.^2.0./(tau_n+ttau))/2.0-sum(log(1.0+ttau./tau_n))/2.0

    p1= sum(log(diag(L))) -sum(lZ) -tnu'*Sigma*tnu/2;
    p2= (nu_n-m.*tau_n)'*((ttau./tau_n.*(nu_n-m.*tau_n)-2*tnu)./(ttau+tau_n))/2;
    p3= sum(tnu.^2./(tau_n+ttau))/2-sum(log(1+ttau./tau_n))/2;
    nlZ = p1-p2+p3;

#println("p1-->");
#println(p1);
#println("p2-->");
#println(p2);
#println("p3-->");
#println(p3);
#println("nlZ-->");
#println(nlZ);
#println("nlZa-->");
#println(nlZa);

    return Sigma,mu,nlZ,L
end
