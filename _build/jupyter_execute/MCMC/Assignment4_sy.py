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

# # Imports

# In[1]:


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


# # Initialization

# In[2]:


# Define initial conditions of SIR model
i0 = 0.01 #fractions infected at time t0=0 (1%)
r0 = 0.00 #fraction of recovered at time t0=0
d0 = 0.00 # fraction of dead at time t0=0
f = 3.0 # time factor, defines total time window range
timestep_data = 0.5 # dt for data (e.g., weekly)
times = np.arange(0,5*f,timestep_data) #np.array from 0 to 15 with steps of 0.5
#ground truth (fixed gamma=1, then R0=beta, time scale to t/gamma)
beta = 2.5
#Mortality Rate
mu = 0.1


# ### SIR Model Nagler

# In[3]:


# ODE SIR system, only parameter p[0]=R0, for scaled time t/gamma (optimized, and 1=gamma)
def SIR(y, t, p):
    #ds = -p[0]*y[0]*y[1] # we do not need susceptibles as S=1-I-R is determ dependent on i and r
    #dr = y[1] #component 0 = recovered, gamma=1 (will be returned directly)
    di = p[0]*(1-y[0]-y[1])*y[1] - y[1] #component 1 = infected, gamma=1 (SIR in one line)
    return [y[1], di] # return r(ecov) and i(nfect)


# In[4]:


# Create SIR curves
y = odeint(SIR, t=times, y0=[r0, i0], args=([beta],), rtol=1e-8) # r0 recovered, i0 infected  at t0

# Observational model for muliplicative noise
yobs = np.random.lognormal(mean=np.log(y[1::]), sigma=[0.20, 0.60]) # noise is multiplicative (makes sense here)
#yobs = np.random.normal(loc=(y[1::]), scale=[0.030, 0.060]) # noise is additive (wrong here)


# In[5]:


# Plotting
# Plot the deterministic curves, and those with multiplicative noise
plt.plot(times[1::],yobs, marker='o', linestyle='none')
plt.plot(times, y[:,0], color='C0', alpha=0.5, label=f'$R(t)$')
plt.plot(times, y[:,1], color ='C1', alpha=0.5, label=f'$I(t)$')
plt.legend()
plt.show()


# ### SIRD Model

# In[6]:


def SIRD(y, t, beta, mu):
    """"This function alters the SIR model from Epidemiology in order to incorporate the deaths 
    caused by the disease.
    
    Parameters:
    ----------
    y: np.array
        This array contains the initial values of r0, i0 an d0. The parameter r0 represents the fraction of 
        recovered people at time t0=0, whereas i0 and d0 representes the fraction of infected and dead 
        people at time t0=0 .
    t: np.array
        This array contains the points in time, where i0, r0 and d0 are modelled.
    beta: float
        The contact rate, more specifically the number of lengthy
        contacts a person has per day.
    mu: float
        The mortaltiy rate.
    """
    # Unpack y to Initial values for the recovered, infected and dead
    R, I, D = y
    # Calculate S
    S = 1 - R - I - D
    # Gamma is set to equal 1
    gamma = 1 
    # Change of Infections
    dIdt = beta[0] * S * I - gamma * I - mu * I
    # Change in Recovered
    dRdt = gamma * I
    # Change in Deaths
    dDdt = mu * I
    # Return Recovered, Infected and Death
    return [dRdt, dIdt, dDdt]


# In[7]:


# Create SIRD curves
y = odeint(SIRD, t=times, y0=[r0, i0, d0], args=([beta],mu), rtol=1e-8) # r0 recovered, i0 infected  at t0

# Observational model for muliplicative noise
sigma_R = 0.20
sigma_I = 0.30
sigma_D = 0.40
sigma = [sigma_R, sigma_I, sigma_D]
yobs = np.random.lognormal(mean=np.log(y[1::]), sigma=sigma) # noise is multiplicative (makes sense here)
#yobs = np.random.normal(loc=(y[1::]), scale=[0.030, 0.060]) # noise is additive (wrong here)


# In[8]:


# Plotting
# Plot the deterministic curves, and those with multiplicative noise
plt.plot(times[1::],yobs, marker='o', linestyle='none')
plt.plot(times, y[:,0], color='C0', alpha=0.5, label=f'$R(t)$')
plt.plot(times, y[:,1], color ='C1', alpha=0.5, label=f'$I(t)$')
plt.plot(times, y[:,2], color ='C2', alpha=0.5, label=f'$D(t)$')
plt.title('$beta$: ' + str(beta) + ' $\mu$: ' + str(mu) + ' $\sigma_R$: ' + str(sigma[0]) + ' $\sigma_I$: ' +str(sigma[1]) + ' $\sigma_D$: ' +str(sigma[2]))
plt.legend()
plt.show()


# ## MCMC

# In[28]:


# ODE system container
sird_model = DifferentialEquation(
    func = SIRD,
    times = np.arange(timestep_data,5*f,timestep_data),#start at t=t_1 (and not t0=0, where log(R=0)=undef)
    mu = 0.1,
    n_states = 3, #r(ecovered) and i(nfected) and d(ead) are states
    n_theta = 2, # beta=R0 and mu only parameter
    t0 = 0 # start from zero
)

# Define and perform MCMC
with pm.Model() as basic_model:

    # Distribution of variances, sigma[0] and sigma[1], some good choice, pos. chauchy  
    sigma = pm.HalfCauchy( 'sigma', 1, shape=2 )

    # Prior: R0 is bounded from below (lower=1), (R0, mu=2, sigma=3)
    R0 = pm.Bound(pm.Normal, lower=1)( 'R0', 2, 3 ) # guess of how R0 distribution looks like = Guassian, mean>1
    
    # Our deterministic curves
    sir_curves = sird_model( y0=[r0, i0, d0], theta=[R0, mu] ) # assume gamma = 1, then beta=R0

    # Likelihood function choice: our sampling distribution for multiplicative noise around the I and R curves
    Y = pm.Lognormal('Y', mu=pm.math.log(sir_curves), sd=sigma, observed=yobs) # variances via sigmas, data=yobs
    
    start = pm.find_MAP()
    
    step = pm.NUTS()
    
    trace = pm.sample(400, step=step, random_seed=44)

# Plot results (takes a while, be patient)
pm.traceplot(trace)
pm.summary(trace).round(2)


# In[25]:


pm.pairplot(trace)


# In[ ]:




