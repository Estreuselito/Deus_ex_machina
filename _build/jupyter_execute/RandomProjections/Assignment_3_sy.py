#!/usr/bin/env python
# coding: utf-8

# # Task
# Based on the program developed in the lecture (SparseRandomProjections), analyze 2 databases of your choice (but not exactly the same digits data as in the lecture) using random projections.
# Study the accuracy (or a score of your choice that makes most sense for your data) as a function of the number of dimensions / features that survived the random projection.
# Try to avoid a zick-zack curve below or around the baseline accuracy curve as your final result for both datasets. At least for one dataset the score is expected to be a smooth-ish curve as a function of the kept number of features. Provide a take-home statement and explain every step.
# You will find that data that is embedded in Eukledian spaces (such as digits) may be more appropriate than data for which Eukledian distances are not an excellent distance measure.

# In[2]:


## Random projections of high-dimensional data
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import datasets

import warnings
warnings.filterwarnings('ignore') # works

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# # Concept of Johnson-Lindenstrauss

# The minimum number of components to guarantee the eps-embedding is given by:
# 
# $$
# n\_components >= \frac{4 * log(n\_samples)}{(\frac{\epsilon^2}{2} - \frac{\epsilon^3}{3})}
# $$

# # Implementation of Random Projections

# In[41]:


def random_projection (classifier, dataset, eps, metric, average):
    # Data Cleaning - Drop n/a
    data = dataset.dropna()
    
    # Split Data and Target
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    
    # Perfom train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    
    # Standardize the Data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    #######################################################################
    # Concept of Johnson-Lindenstrauss
    n = data.shape[0]
    # Print what the theory says for k, given an eps(ilon)
    print ("Professors Johnson and Lindenstrauss say: k >=", johnson_lindenstrauss_min_dim(n,eps=eps))
    
    #######################################################################
    # Classification
    # Initialize the model
    model = classifier

    # Train the Model
    model.fit(X_train, y_train)

    # Determine the baseline Score
    if metric == 'Accuracy': 
        baseline = metrics.accuracy_score(model.predict(X_test), y_test)
    else:
        baseline = metrics.f1_score(model.predict(X_test), y_test, average = average)

    # Create empty list to store the performance results
    results = []

    # determine the number of features in the dataset
    m = data.shape[1]
    
    # Create an evenly spaced list
    dims = np.int32(np.linspace(2, m, int(m/3)))
    
    # Loop over the projection sizes, k
    for dim in dims:
        # Create random projection
        sp = SparseRandomProjection(n_components = dim)
        X_train_transformed = sp.fit_transform(X_train)

        # Train classifier of your choice on the sparse random projection
        model = classifier
        model.fit(X_train_transformed, y_train)

        # Evaluate model and update accuracies
        X_test_transformed = sp.transform(X_test)
        if metric == 'Accuracy': 
            results.append(metrics.accuracy_score(model.predict(X_test_transformed), y_test))
        else:
            results.append(metrics.f1_score(model.predict(X_test_transformed), y_test, average = average))

    #######################################################################
    # Plotting
    # Create figure
    plt.figure()
    plt.title('Classifier: ' + str(classifier))
    plt.xlabel("# of dimensions k")
    plt.ylabel(metric)
    plt.xlim([2, m])
    plt.ylim([0, 1])
 
    # Plot baseline and random projection accuracies
    plt.plot(dims, [baseline] * len(results), color = "r")
    plt.plot(dims, results)

    plt.show()


# ### Application and Analysis

# URLs for Different Datasets:
# 
# **Wine Quality (Red):**
# https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv  
# **Default of credit card clients:**
# https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls

# In[48]:


# load dataset
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'

if data_url[-4:] == '.csv':
    df_dataset = pd.read_csv(data_url, sep=';')
if data_url[-4:] == '.xls':
    df_dataset = pd.read_excel(data_url, header = 1) # header = 1 to indicate that the first row is the column name

print('The class label frequency is:\n', df_dataset.iloc[:, -1].value_counts())

print('The shape of the dataset is:\n', df_dataset.shape)


# In[59]:


# Determine Classifier
classifier = SVC(kernel='rbf', random_state = 0)

# Determine eps
eps = 0.1

# Determine performance metric
metric = 'f1'

# Determine average for score evaluation
# for binary classification use 'binary', for mutliclass use 'weighted'
average = 'binary'

# Call function
random_projection(classifier, df_dataset[:5000], eps, metric, average)


# # <span style="color:red"> Old Code for GridSearchCV - To be deleted if not needed at a later stage </span>

# In[ ]:


def preprocessing (dataset):
    # Data Cleaning - Drop n/a
    data = dataset.dropna()
    
    # Split Data and Target
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    
    # Perfom train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    
    # Standardize the Data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test


# ## Initial GridSearch to find best parameters

# In[ ]:


# load dataset
df_dataset = pd.read_csv('XXX')

# Preprocessing and Train-Test-Split
X_train, X_test, y_train, y_test = preprocessing(df_dataset[:10000])

# Perform GridSearchCV
model = SVC(random_state=0)
# param_grid = {'kernel' : ['poly', 'rbf', 'linear', 'sigmoid'],
              # 'C' : [1, 10, 25, 50, 75, 100]}
              # 'gamma' : [0.01, 0.1, 1],
              # 'degree' : [2, 3, 4, 5, 6, 7]}
            
param_grid = {'kernel' : ['poly', 'rbf'],
              'C' : [1, 10, 25, 50, 75, 100]}
              # 'gamma' : [0.01, 0.1, 1],
              # 'degree' : [2, 3, 4, 5, 6, 7]}

scoring = {'f1' : 'f1'}

CV_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring = scoring, refit = 'f1', cv=5)

CV_model.fit(X_train, y_train)

best_params = CV_model.best_params_
print('Best Parameters:\n', best_params)


# In[ ]:




