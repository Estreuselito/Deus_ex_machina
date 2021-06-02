#!/usr/bin/env python
# coding: utf-8

# In[118]:


import numpy as np
import scipy.stats as stats


# In[119]:


x = np.random.randint(0, 100, 100)


# In[120]:


y = np.random.randint(0, 86, 100)


# In[121]:


z = np.linspace(0, 200, 100)


# In[124]:


x = stats.exp.pdf(x, loc=0, scale=5)
# x = stats.uniform.pdf(x, loc=0, scale=5)


# In[126]:


# sample size of data we observe, trying varying this (keep it less than 100 ;)
N = 1

# the true parameters, but of course we do not see these values...
lambda_1_true = 1
lambda_2_true = 3

#...we see the data generated, dependent on the above two values.
data = np.concatenate([
    stats.poisson.rvs(lambda_1_true, size=(N, 1)),
    stats.poisson.rvs(lambda_2_true, size=(N, 1))
], axis=1)


# In[127]:


x = y = np.linspace(.01, 5, 100)
likelihood_x = np.array([stats.poisson.pmf(data[:, 0], _x)
                        for _x in x]).prod(axis=1)
likelihood_y = np.array([stats.poisson.pmf(data[:, 1], _y)
                        for _y in y]).prod(axis=1)
L = np.dot(likelihood_x[:, None], likelihood_y[None, :])


# In[131]:


L.flatten()


# In[143]:


import matplotlib.pyplot as plt
import numpy as np

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

def plot(i):

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    theta = 2 * np.pi * np.random.random(1000)
    r = i * np.random.random(1000)
    x = np.ravel(r * np.sin(theta))
    y = np.ravel(r * np.cos(theta))
    z = f(x, y)

    ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
    fig.tight_layout()

plot(6)
# interactive_plot


# In[132]:


# Make the plot
fig = plt.figure(figsize=(12,12))
ax = plt.axes(projection='3d')
ax.plot_trisurf(L.flatten(), y, z, cmap=plt.cm.viridis, linewidth=0.2)
plt.show()


# In[17]:


df["Z"].value_counts()
# df["Y"].max()


# In[1]:


# libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
 
# Get the data (csv file is hosted on the web)
url = 'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/volcano.csv'
data = pd.read_csv(url)
 
# Transform it to a long format
df=data.unstack().reset_index()
df.columns=["X","Y","Z"]
 
# And transform the old column name in something numeric
df['X']=pd.Categorical(df['X'])
df['X']=df['X'].cat.codes
 
# Make the plot
fig = plt.figure(figsize=(12,12))
ax = fig.gca(projection='3d')
ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
plt.show()
 
# to Add a color bar which maps values to colors.
fig = plt.figure()
ax = fig.gca(projection='3d')
surf=ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
fig.colorbar( surf, shrink=0.5, aspect=5)
plt.show()
 
# Rotate it
fig = plt.figure()
ax = fig.gca(projection='3d')
surf=ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
ax.view_init(30, 45)
plt.show()
 
# Other palette
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.jet, linewidth=0.01)
plt.show()


# In[7]:


# MCMC introduction
# Based on some PyMC3 docu examples 
# Jan Nagler, May, 2021, A.C.

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
# set output floats to 5 significant digits throughout the journey
get_ipython().run_line_magic('precision', '4')
get_ipython().run_line_magic('matplotlib', 'inline')
# This is a massive lib for MCMC 
import pymc3 as pm
print('Running on PyMC3 v{}'.format(pm.__version__))
# pip install git+https://github.com/pymc-devs/pymc3
# just run this commend in your shell
np.random.seed(42)
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# In[8]:


# Model: Simple random additive noise as two features

# Groundtruth parameter values
alpha, sigma = 1, 1 # intercept and std
beta = [1, 2.5] # weights

# Size of surrogate dataset
size = 25 # keep this LOW when getting started, increase when you and your CPU are ready

# Predictor variable
X1 = np.random.randn(size) * 2
X2 = np.random.randn(size) * 0.3

# Surrogate data set
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma


# In[9]:


#plot data, projections
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,4))
axes[0].scatter(X1, Y)
axes[1].scatter(X2, Y)
axes[0].set_ylabel('Y'); axes[0].set_xlabel('X1'); axes[1].set_xlabel('X2');


# Define prior distributions for the parameters for which we will be trying to infer posterior distributions

# In[10]:


# Build a model 
basic_model = pm.Model()

with basic_model: 

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Expected value of outcome
    # Determinitic random variable, means sharply defined by delta function over data and model param
    mu = alpha + beta[0]*X1 + beta[1]*X2 

    # Likelihood (sampling distribution) of observations, given out surrogate data Y
    # Called observed stochastic variable 
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)


# Perform MCMC: Sample from our prior distribution, simulate the model, and evaluate the likelihood of the data given those input parameters, based on our "basic model", and the "noise" distribution, our (surrogate) data Y. Use Bayesian inference to decide whether to keep the trial points or throw them away. Choose a new set of points and start over.

# In[11]:


# MCMC Sampling method 
with basic_model:
    
    # Compute MAP estimate
    start = pm.find_MAP() # args, e.g., model=basic_model, Powell method (no derivatives needed)
    print("Start", start)
    
    # instantiate sampler steps: Optional 
    #step = pm.Slice()
    #step = pm.Metropolis()
    step = pm.NUTS()
    
    # draw 5000 posterior samples
    trace = pm.sample(500, step=step)


# In[13]:


pm.traceplot(trace)
#pm.autocorrplot(trace)
pm.pairplot(trace)
#pm.energyplot(trace) # https://arxiv.org/pdf/1604.00695v1.pdf
pm.summary(trace).round(2)


# In[ ]:





# In[ ]:




