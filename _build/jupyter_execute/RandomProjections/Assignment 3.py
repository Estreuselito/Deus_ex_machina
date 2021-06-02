#!/usr/bin/env python
# coding: utf-8

# \begin{titlepage}
# 
# % Photo of FS
# \centering
# \vspace{-40pt}
# \includegraphics[width=0.8\textwidth]{Frankfurt_School_Logo.jpg}\par
# \vspace{2.5cm}
# 
# % Course
# {\scshape\huge Assignment 3 \par}
# \vspace{2.5cm}
# 
# % Title
# {\Huge\bfseries Sparse Random Projection \par}
# {\scshape\large Jan's birthday edition \par}
# 
# \vspace{2cm} % If signature is taken might have to add space.
# 
# 
# {\Large Yannik Suhre \par}
# {\Large Skyler MacGowan \par}
# {\Large Debasmita Dutta \par}
# {\Large Sebastian Sydow \par}
# \vspace{0.5cm}
# 
# % Date
# \vfill
# {\large \today\par}
# \end{titlepage}
# 
# 
# \newpage
# 
# \hypersetup{linkcolor=black}
# \tableofcontents
# 
# \newpage

# # The Johnson-Lindenstrauss Lemma
# The Johnson-Lindenstrauss (JL) Lemma is the math behind Euclidean Distance/Space; it is what proves the “*approximate maintenance of distance between the data points in different dimensions*” property to be true. The lemma states that a small set of points in a high-dimensional space can be embedded into a space of much lower dimension in such a way that distances between the points are nearly preserved. The function `johnson_lindenstrauss_min_dim` of [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.johnson_lindenstrauss_min_dim.html)
# calculates the  minimum number of components $k$, i.e. the number of dimensions in which distances between 
# the points are nearly preserved, by the following formula:
# 
# \begin{equation}
# k >= \frac{4 * log(n\_samples)} {(\frac{\epsilon^2}{2} - \frac{\epsilon^3}{3})}
# \end{equation}
# 
# Nevertheless, this only holds if the correct $k$ dimensions are chosen and not just any $k$ dimensions.
# 
# The following outlines the key components of the JL Lemma and what they represent. 
# 
# - $k$: This represents the minimum number of dimensions to which the dataset can be reduced to without a substantive decrease in accuracy, down from the original *d* dimensions. This *k* is in effect the result obtained from the JL Lemma formula, based on the parameters provided thereto. 
# - $\epsilon$: This represents the error term. Namely, in the context of conducting random projections in Euclidean space it is the **approximate** distance that is maintained, i.e. there is some error involved during this process. Naturally, that error could result in either an increase or a decrease in the distance, which is portrayed by the inequality below. In this inequality, 1 represents the original distance, $\epsilon$ represents the error the user is willing to accept (range from 0 to 1, with lower values indicating a lower tolerance for error), the superscript indicates that this inequality applies to Euclidean Space (and hence the L2 norm), the distance of the transformed vectors is portrayed by the middle term ($f(x_i) - f(x_j)$), while the outer terms represent the two possible boundaries, i.e. that the new distance is somewhat smaller or greater by $\epsilon$ than the original distance.
# 
# \begin{equation}
# (1 - \epsilon) ||x_i - x_j||^2_2 \leq ||f(x_i) - f(x_j)||^2_2 \leq (1 + \epsilon) ||x_i - x_j||^2_2
# \end{equation}
# 
# - $n\_samples$: This refers to the number of observations in the dataset.
# 
# **Note**: The number of dimensions is independent of the original number of features but instead depends on the size of the dataset: the larger the dataset, the higher is the minimal dimensionality of an $\epsilon$-embedding.
# 
# **Note**: In the context of dimensionality reduction, random projections are typically used when one is unable to reliably calculate the covariance matrix (due to data sparsity for example), whereas when the covariance can be reliably calculated then **P**rincipal **C**omponent **A**nalysis (PCA) is used. Both PCA and random projections require the dataset to be in Euclidean Space in order to function properly (more on Euclidean Space below). PCA is computionally more expensive which also factors into the choice of dimensionalty reduction method selected.

# # Euclidean Space/Data
# 
# When a given dataset is said to be in "Euclidean Space", that means that the distance between the observations in the dataset is linearly defined. Essentially, you can draw a line between each vector pair, and this line represents the distance between each pair. Distance, in turn, is a measure of similarity, with lesser distances indicating greater similarity and vice versa. 
# 
# How is one to know whether a given dataset is in Euclidean Space? Well essentially anything embedded in physical space could reliaby said to be in Euclidean Space, because in such circumstances one can draw a line between two points and trust that that accurately represents the distance or similarity between them. In a geographical context for example, the distance *as the crow flies* between the Frankfurt School to the Abdeen Palace Museum in Cairo is 2,922.37 km whereas that between the Frankfurt School and the Church of the Holy Sepulchre in Jerusalem is 2,993.46 km; these are examples of euclidean distances, and from them we can determine that Frankfurt School is 71.12 km closer to the Abdeen Palace Museum than it is to the Church of the Holy Sepulchre. 

# # Non-Euclidean Spaces/Data
# 
# In Non-Euclidean Space, the "linearity" property described above in reference to Euclidean space does not hold; one cannot simply draw a line between the constituent vector pairs and trust that this is an accurate measure of the similarity thereof (it isn't). Instead, in non-euclidean spaces the degree of similarity of the vectors should be measured using another (non-linear) scale, e.g. logarithmic, exponential, etc. 
# 
# For example, we define the loudness of a given sound via the decibel (dB) measure. In this measure, an increase of three decibels corresponds to a doubling of the overall loudness. Another fairly well-known example would be the moment-magnitude scale, which is the principal measure now used when assessing the strength and destructive potential of earthquakes. This scale goes from one to ten, with each step representing a 32 times larger release of energy than the preceeding step. For example, a 8.0 earthquake ("Great", occurs roughly once a year) releases 31,623 times as much energy as does a 5.0 earthquake ("Moderate", occurs roughly 1250 times per year).
# 
# United States Geographical Survey Earthquake Magnitude Comparison Calculator: https://earthquake.usgs.gov/education/calculator.php

# In[1]:


# Group 10 - Skyler MacGowan, Sebastian Sydow, Debasmita Dutta, Yannik Suhre
from sklearn.svm import LinearSVC, SVC
from RandomProjectionClass import RandomSparseRepresentation
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# # Introduction to the Datasets
# 
# For this assignment, two datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php.) were chosen. Below is an overview of these datasets.  
# 
# ## Classification of Frogs
# The dataset is used to classify the species of frogs. As the dataset encompasses ten different classes, i.e. `Species`, it is a multi-class classification. The dataset contains 7,195 observations and 23 features, one of which is the target-feature (`Species`, after deleting the columns `Family`, `Genus` and `RecordID`). It is an imbalanced dataset, as some classes are much more frequent than others. Thus, the performance of the classifier will be evaluated using the `f1_score` with the parameter settings `average = weighted`. This dataset is already standardized.  
# 
# Data Source: https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29
# 
# ## Classification of Dry Beans  
# The dataset is used to classify dry beans into seven different classes; it is a multi-class classification. The dataset contains 13,611 observations and 17 features, one of which is the target-feature (`Class`).As with the first dataset, the performance of the classifier will be evaluated using the `f1_score` with the parameter settings `average = weighted` as this dataset is also imbalanced. This dataset is not yet standardized.
# 
# Data Source: https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset

# In[11]:


data = RandomSparseRepresentation(birthday_version=False)


# In[12]:


data.get_data("./data/Frogs_MFCCs.csv",
              data_type = ".csv")


# In[13]:


data.split_data(standardize = False, columns_to_drop = ["RecordID", "Family", "Genus"])


# In[14]:


data.JL_lemma()


# In[6]:


data.baseline(model = SVC, kernel='rbf', gamma = 0.1, C=5, random_state = 0)


# In[7]:


data.apply_random_projection(model = SVC, kernel='rbf', gamma = 0.1, C=5, random_state = 0)


# In[15]:


dry_beans = RandomSparseRepresentation(text = False)


# In[16]:


# Plot explained variances
dry_beans.prepare_fit(url = "./data/Dry_Bean_Dataset.xlsx", data_type = ".xlsx",
                     standardize = True,
                     model = SVC, kernel='rbf', gamma = 0.1, C=5, random_state = 0)


# In[10]:


font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 13}

plt.rc('font', **font)

fig, ax = plt.subplots(1,2, figsize = (16,10))
ax[0].set_ylim([0,1])
ax[0].plot(data.dims, [data.baseline] * len(data.accuracies), color = "r")
ax[0].plot(data.dims, data.accuracies)
ax[0].set_title("Sparse Random Projection with Frog data")
ax[0].set_xlabel('# of dimensions')
ax[0].set_ylabel(f"{data.metric}")
ax[1].plot(dry_beans.dims, [dry_beans.baseline] * len(dry_beans.accuracies), color = "r")
ax[1].plot(dry_beans.dims, dry_beans.accuracies)
ax[1].set_title("Sparse Random Projection with Dry Beans Data")
ax[1].set_ylim([0,1])
ax[1].set_xlabel('# of dimensions')
ax[1].set_ylabel(f"{dry_beans.metric}")
plt.show() 

Sparse Random Projection:
    Get dataset
    n = 1
    Redo:
        random linear procjection into n dimensions
        evaluate the score
        n + 1
   Display n
   Choose n where it almost converges with the baseline metric score
# # Takeaway to go
# As can be inferred from both graphs above, the random dimensionality reduction worked for both datasets, even for smaller dimensions than those obtained from the JL Lemma. In both cases, we see how initially the performance following a reduction in dimensionality stays at or just slightly below the baseline. At a certain point however, the new performance drops off substantially, again in both datasets. This point (where the new performance decreases substantially), represents the number of dimensions that should be maintained when conducting the dimensionality reductions. Furthermore, both classifiers perform very well as can be seen by the high `f1_score`s.
# 
