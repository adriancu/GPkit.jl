### Some examples to try things out - as a repl session

# note, for now NLopt.jl is a dependency of GPkit. Add it with Pkg.add("Nlopt.jl")

include("src/GPkit.jl") # if you've cloned the repo
using GPkit

julia> using GPkit

# a cannonical model
x=[0.3,0.6,1.5,1.6]
y=[0.2,0.4,0.9,0.7]
xs=[0.8,1.2]

# some lower level stuff to try the basics and compare intermediate steps
# against the gpml.
n=length(x)
covfn=CovSEiso(0.5,0.8)   # instantiate the covariance function
sn2=0.01^2                # the model likelihood (signal noise)
mu=fill!(similar(x),0.0)  # won’t use the mean function
C=doCov(covfn,x,x)        # compute C the covariance matrix
L=chol(C+(eye(n)*sn2),:L) # obtain Cholesky decomposition (C==L’L)
alpha=L'\(L\y)
Cs=doCov(covfn,x,xs)      # C_star
fs=Cs'*alpha              # f_star -> predictive means
v=L\Cs                    # # solve for v
Css=doCov(covfn,xs,[])    # C_star_star
#Vfs=Css-v'*v              # V_star -> predictive variances
#Vmdl=Vfs+sn2              # With signal noise (Lik)

# Some examples using the julia type and showing off multiple dispatch
# use same X,y, Xs from above
cov=CovSEiso(0.5,0.8);    # use ; to suppress return being printed in repl
lik=LikGauss(0.05);
gp=GPmodel(InfExact(), cov, lik, MeanZero(), x, y);
# now we can run the model - you'll get deprecation warnings in julia 0.4,
# use julia --depwarn=no when starting the repl to supporess them
(post,nlZ,dnlZ)=inference(gp, with_dnlz=false); # posterior and derivatives if requested
(ymu,ys2,fmu,fs2,lp)=prediction(gp, post, xs);  # predictions and variances (see gpml)
# ymu ==>
#2x1 Array{Float64,2}:
# 0.646651
# 1.07101
# ys2 ==>
#2x1 Array{Float64,2}:
# 11.9323
# 12.4737


# now we can do some optimisation...
#using NLopt # after doing Pkg.add("Nlopt.jl")
# See docs for NLopt.jl as well
(optf,optx,ret) = optinf(gp, 200, algo=:LD_LBFGS, with_dnlz=true); # optf has new hypers

(ymu,ys2,fmu,fs2,lp)=prediction(gp, post, xs);  # optimised hypers were "left" in gp.covfn and gp.likfn

# optimised ymu ==>
#2x1 Array{Float64,2}:
# 0.425662
# 0.433371
#optimised ys2 ==>
#2x1 Array{Float64,2}:
# 6.26909
# 6.17062

############ some actual documentation still to come.
