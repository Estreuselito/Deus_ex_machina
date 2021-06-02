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
!pip install pymc3!pip install Theano[test]
# 

# In[8]:


# Assignment 4: SIR model, MCMC for R0
# Jan Nagler
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


# In[9]:


# Define initial conditions of SIR model
i0 = 0.01 #fractions infected at time t0=0 (1%)
r0 = 0.00 #fraction of recovered at time t0=0
f = 3.0 # time factor, defines total time window range
timestep_data = 1 # dt for data (e.g., weekly)

# ODE SIR system, only parameter p[0]=R0, for scaled time t/gamma (optimized, and 1=gamma)
def SIR(y, t, p):
    #ds = -p[0]*y[0]*y[1] # we do not need susceptibles as S=1-I-R is determ dependent on i and r
    #dr = y[1] #component 0 = recovered, gamma=1 (will be returned directly)
    di = p[0]*(1-y[0]-y[1])*y[1] - y[1] #component 1 = infected, gamma=1 (SIR in one line)
    return [y[1], di] # return r(ecov) and i(nfect)

times = np.arange(0,5*f,timestep_data)

#ground truth (fixed gamma=1, then R0=beta, time scale to t/gamma)
beta = 1.7

# Create SIR curves
y = odeint(SIR, t=times, y0=[r0, i0], args=([beta],), rtol=1e-8) # r0 recovered, i0 infected  at t0

# Observational model for muliplicative noise
yobs = np.random.lognormal(mean=np.log(y[1::]), sigma=[0.20, 0.60]) # noise is multiplicative (makes sense here)
#yobs = np.random.normal(loc=(y[1::]), scale=[0.030, 0.060]) # noise is additive (wrong here)

# Plot the deterministic curves, and those with multiplicative noise
plt.plot(times[1::],yobs, marker='o', linestyle='none')
plt.plot(times, y[:,0], color='C0', alpha=0.5, label=f'$R(t)$')
plt.plot(times, y[:,1], color ='C1', alpha=0.5, label=f'$I(t)$')
plt.legend()
plt.show()


# In[11]:


import os
# one of
os.environ['MKL_THREADING_LAYER'] = 'sequential'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'


# In[12]:


# ODE system container
sir_model = DifferentialEquation(
    func = SIR,
    times = np.arange(timestep_data,5*f,timestep_data), #start at t=t_1 (and not t0=0, where log(R=0)=undef)
    n_states = 2, #r(ecovered) and i(nfected) are states
    n_theta = 1, # beta=R0 only parameter
    t0 = 0 # start from zero
)

# Define and perform MCMC
if __name__ == "__main__":
    with pm.Model() as basic_model:

        # Distribution of variances, sigma[0] and sigma[1], some good choice, pos. chauchy  
        sigma = pm.HalfCauchy( 'sigma', 1, shape=2 )

        # Prior: R0 is bounded from below (lower=1), (R0, mu=2, sigma=3)
        R0 = pm.Bound(pm.Normal, lower=1)( 'R0', 2, 3 ) # guess of how R0 distribution looks like = Guassian, mean>1

        # Our deterministic curves
        sir_curves = sir_model( y0=[r0, i0], theta=[R0] ) # assume gamma = 1, then beta=R0

        # Likelihood function choice: our sampling distribution for multiplicative noise around the I and R curves
        Y = pm.Lognormal('Y', mu=pm.math.log(sir_curves), sd=sigma, observed=yobs) # variances via sigmas, data=yobs

        start = pm.find_MAP()

        step = pm.NUTS()

        trace = pm.sample(40, step=step, random_seed=44, cores=3)

# Plot results (takes a while, be patient)
pm.traceplot(trace)
pm.summary(trace).round(2)


# In[ ]:


pm.pairplot(trace)


# In[ ]:




