#!/usr/bin/env python
# coding: utf-8

# # Surrogate data
# 
# The package `sklearn` has a module, which is called `datasets`. Within this package there are ~30 functions, which load sample data. All of those functions start with `load_...`. Moreover, this package has ~20 sample generators. These sample generators create surrogate data, which is different seperated based on the function. They all start with `make_...`. For a more exhaustive description of these functions please review the `sklearn` documentation found in bibliography [1].
# 
# The first step is to import that module from sklearn. We will also import matplotlib to plot what we have just created.

# In[18]:


from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


# ## `make_blobs`
# 
# The first function we can use to create surrogate data is the function called `make_blobs`. This creates Gaussian blobs for clustering. With the parameter `n_samples` one can influence the total number of points generated, with the `centers` parameter one can set the number of centers, viz. different classes for a classification problem, and with the `cluster_std` one can set the standard deviation around each center (i.e. as higher the `cluster_std` as higher the "noise" around the centers).
# 
# The function returns two values. Once the `X` variable, which contains the different datapoints, and once the `y` which contains the different labels for the datapoints. Over that we can check later how good/bad the cluster algorithm performed.

# In[25]:


X, y = datasets.make_blobs(n_samples = 1000, centers = 5, cluster_std = 0.5)
for cluster in np.unique(y):
    plt.scatter(X[:,0][y==cluster], X[:,1][y==cluster])
plt.title("Make blobs function")
plt.show()


# ## `make_circles`
# 
# The second function is called `make_circles`, which returns a circle within yet another circle, as one can see below. This function as well has different classes, as can be seen by the different colors. For this exercise we will use the parameters `n_samples`, which works like with the `make_blobs` function, and the `noise` parameter, which works like the `cluster_std` from `make_blobs`.  

# In[26]:


X, y = datasets.make_circles(n_samples = 1000, noise = 0.01)
for cluster in np.unique(y):
    plt.scatter(X[:,0][y==cluster], X[:,1][y==cluster])
plt.title("Make circles function")
plt.show()


# ## `make_classification`
# 
# Next, we will briefly introduce the `make_classification` function from `sklearn`s `datasets`. This function creates a random n-classification problem in a normal distribution. Hence, we can use the parameter `n_classes` to change the number of classes. I will set it to `5`, but that can be altered. With the parameter `n_informative` (which has to be >= `n_classes`) defines the number of informative features in the generated dataset.

# In[45]:


X, y = datasets.make_classification(n_samples=1000, n_classes = 5, n_informative = 10)
for cluster in np.unique(y):
    plt.scatter(X[:,0][y==cluster], X[:,1][y==cluster])
plt.title("Make classification function")
plt.show()


# ## `make_hastie_10_2`
# 
# Another function we want to test our algorithm on is the `make_hastie_10_2` function. This reproduces the example 10.2 from Haste et al. 2009 [2]. It defines the clusters by the following algorithm:
# ```python
# y[i] = 1 if np.sum(X[i] ** 2) > 9.34 else -1
# ```
# The only parameter which can be altered here are the `n_samples`, viz. the number of samples. 

# In[49]:


X, y = datasets.make_hastie_10_2(n_samples=1000)
for cluster in np.unique(y):
    plt.scatter(X[:,0][y==cluster], X[:,1][y==cluster])
plt.title("Make Hastie 10.2 function")
plt.show()


# ## `make_moons`
# 
# This function creates two interleaving half circles. It takes the arguments `n_samples` and `noise`, just like the function `make_circles`.

# In[52]:


X, y = datasets.make_moons(n_samples = 1000, noise = 0.01)
for cluster in np.unique(y):
    plt.scatter(X[:,0][y==cluster], X[:,1][y==cluster])
plt.title("Make moons function")
plt.show()


# ---

# # Bibliography
# 
# [1] https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
# 
# *Note: This is the exhaustive documentation of all available datasets and samples generator webpage. It offers deeper insights into which parameters else each function has.*
# 
# [2] T. Hastie, R. Tibshirani and J. Friedman, “Elements of Statistical Learning Ed. 2”, Springer, 2009.
# 
# *Note: This reference was directly taken from `sklearn`s documentation!* 
