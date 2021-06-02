#!/usr/bin/env python
# coding: utf-8

# # https://stackoverflow.com/questions/67035772/pymc3-using-start-value-for-lkjcholeskycov-gives-bad-initial-energy-error

# In[2]:


# Working example:
import numpy as np
import pymc3 as pm
n_samples = 20
n_tune_samples = 10

mu = np.zeros(3)
true_cov = np.array([[1.0, 0.5, 0.1],
                     [0.5, 2.0, 0.2],
                     [0.1, 0.2, 1.0]])
data = np.random.multivariate_normal(mu, true_cov, 10)
print(data.shape)


with pm.Model() as model1:
    sd_dist = pm.Exponential.dist(1.0, shape=3)
    print(sd_dist.shape)

    chol, corr, stds = pm.LKJCholeskyCov('chol_cov', n=3, eta=2,
        sd_dist=sd_dist, compute_corr=True)
    vals = pm.MvNormal('vals', mu=mu, chol=chol, observed=data)

with model1:
    trace1 = pm.sample(draws=n_samples, tune=n_tune_samples)


# In[8]:


np.diag([0.56, 0.61, 0.74])[np.tril_indices(3)]


# In[6]:


n_samples = 20
n_tune_samples = 10

mu = np.zeros(3)
true_cov = np.array([[1.0, 0.5, 0.1],
                     [0.5, 2.0, 0.2],
                     [0.1, 0.2, 1.0]])
data = np.random.multivariate_normal(mu, true_cov, 10)
print(data.shape)

with pm.Model() as model2:
    sd_dist = pm.Exponential.dist(1.0, shape=3)
    print(sd_dist.shape)
    
    chol, corr, stds = pm.LKJCholeskyCov('chol_cov', n=3, eta=2,
        sd_dist=sd_dist, compute_corr=True)
    
    vals = pm.MvNormal('vals', mu=mu, chol=chol, observed=data)

chol_init = np.diag([0.56, 0.61, 0.74])[np.tril_indices(3)]
with model2:
    
       
    trace2 = pm.sample(draws=n_samples, tune=n_tune_samples,
                       start={'chol_cov':chol_init})


# In[ ]:




