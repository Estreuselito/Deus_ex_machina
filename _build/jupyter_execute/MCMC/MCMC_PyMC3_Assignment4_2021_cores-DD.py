#!/usr/bin/env python
# coding: utf-8

# Assignment 4 (Due: Wed, May 12, 2021)
# 
# The devised program estimates robustly, given very noisy and very sparse data of infected and recovered of a past epidemic, the basic reproduction number of the SIR model. To keep computation in limit, we assume gamma=1. The SIR model is implemented in a minimal and optimal way, using scaled variables and a scaled time. Only the ODE part is numerically integrated that needs to be integrated. The noisy number of infected and the number of recovered are highly correlated. This relationship helps MCMC infer the parameters. 
# 
# Get familiar with the commented MCMC code below.
# 
# Task:
# Change the program to the SIRD model, by including (D)eaths, with rate $\mu$. Fix not only $\gamma=1$ but also $\beta=2.5$ (or to a higher value of your choice). 
# Infer the death rate $\mu$, given noisy $S(t)$, $I(t)$, $R(t)$, $D(t)$ input curves.
# If you want, you can try to optimze the code (optional, very very hard). 
# Also optional is: Does the inference for $\mu$ work, if $S(t)$ and/or $R(t)$ are not given ?
# You may use these (initial) conditions/parameters: $$i0 = 0.01, s0 = 0.99, r0 = 0, d0 = 0, f = 3.0, timestep = 0.5.$$
# You may assume values for the respective $\sigma$'s (log-normal noises) in the range of $0.2-0.4$, but not lower than $0.1$. Good luck and have fun.

# In[1]:


# Assignment 4: SIR model, MCMC for R0
# Jan Nagler
# based on https://docs.pymc.io/pymc-examples/examples/ode_models/ODE_API_introduction.htm
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('precision', '4')
get_ipython().run_line_magic('matplotlib', 'inline')
import pymc3 as pm #install if necessary
from pymc3.ode import DifferentialEquation
from scipy.integrate import odeint
import warnings
warnings.filterwarnings("ignore") 
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
np.random.seed(42)


# In[2]:


# Define initial conditions of SIR model
i0 = 0.01 #fractions infected at time t0=0 (1%)
r0 = 0.00 #fraction of recovered at time t0=0
d0 = 0.00
f = 3 #3.0 # time factor, defines total time window range
timestep_data = 0.5 # dt for data (e.g., weekly)
# d is mu
# ODE SIR system, only parameter p[0]=R0, for scaled time t/gamma (optimized, and 1=gamma)
def SIRD(y, t, p):
    #ds = -p[0]*y[0]*y[1] # we do not need susceptibles as S=1-I-R is determ dependent on i and r
    #dr = y[1] #component 0 = recovered, gamma=1 (will be returned directly)
    di = p[0]*(1-y[0]-y[1])*y[1] - y[1]-p[1]*y[1] #component 1 = infected, gamma=1 (SIR in one line)
    dd = p[1]*y[1]
    return [y[1],di,dd] # return r(ecov) and i(nfect)

times = np.arange(0,5*f,timestep_data)

#ground truth (fixed gamma=1, then R0=beta, time scale to t/gamma)
beta = 2.5
d = 0.1
p = [beta,d]
print(p)

# Create SIR curves
y = odeint(SIRD, t=times, y0=[r0, i0,d0], args=([p[0],p[1]],), rtol=1e-8) # r0 recovered, i0 infected  at t0
# Observational model for muliplicative noise
yobs = np.random.lognormal(mean=np.log(y[1::]), sigma=[0.2, 0.3, 0.4]) # noise is multiplicative (makes sense here)
#yobs = np.random.normal(loc=(y[1::]), scale=[0.030, 0.060]) # noise is additive (wrong here)

# Plot the deterministic curves, and those with multiplicative noise
plt.plot(times[1::],yobs, marker='o', linestyle='none')
plt.plot(times, y[:,0], color='C0', alpha=0.5, label=f'$R(t)$')
plt.plot(times, y[:,1], color ='C1', alpha=0.5, label=f'$I(t)$')
plt.plot(times, y[:,2], color ='C2', alpha=0.5, label=f'$D(t)$')
plt.legend()
plt.show()


# In[3]:


# ODE system container
sir_model = DifferentialEquation(
    func = SIRD,
    times = np.arange(timestep_data,5*f,timestep_data), #start at t=t_1 (and not t0=0, where log(R=0)=undef)
    n_states = 3, #r(ecovered) and i(nfected) are states
    n_theta = 2, # beta=R0 only parameter
    t0 = 0 # start from zero
)

# Define and perform MCMC
with pm.Model() as basic_model:

    # Distribution of variances, sigma[0] and sigma[1], some good choice, pos. chauchy  
    sigma = pm.HalfCauchy( 'sigma', 1, shape=3 )

    # Prior: R0 is bounded from below (lower=1), (R0, mu=2, sigma=3)
    #R0 = pm.Bound(pm.Normal, lower=1)( 'R0', 2, 3 ) # guess of how R0 distribution looks like = Guassian, mean>1
    beta_prior = pm.Bound(pm.Normal, lower=1)( 'beta', 2, 3 ) # guess of how R0 distribution looks like = Guassian, mean>1
    d_prior = pm.Bound(pm.Normal, lower=1)( 'd', 2, 3 ) # ??? no idea
    
    # Our deterministic curves
    sir_curves = sir_model( y0=[r0, i0,d0], theta=[beta_prior,d_prior] ) # assume gamma = 1, then beta=R0

    # Likelihood function choice: our sampling distribution for multiplicative noise around the I and R curves
    Y = pm.Lognormal('Y', mu=pm.math.log(sir_curves), sd=sigma, observed=yobs) # variances via sigmas, data=yobs
    
    start = pm.find_MAP()
    
    step = pm.NUTS()
    
    trace = pm.sample(10, step=step, cores=1, random_seed=44) #set here number of cores, to adapt for hardware

# Plot results (takes a while, be patient)
pm.traceplot(trace)
pm.summary(trace).round(2)


# In[4]:


pm.pairplot(trace)


# In[6]:


print(start)


# In[11]:


print(trace['beta'][-10:])
print(trace['d'][-10:])
print(trace['sigma'][-10:])


# In[ ]:




