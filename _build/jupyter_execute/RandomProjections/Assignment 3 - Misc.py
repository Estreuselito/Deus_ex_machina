#!/usr/bin/env python
# coding: utf-8

# ## The Johnson-Lindenstrauss Lemma
# ### NOTE: THIS SECTION WAS TRANSFERRED TO THE MASTER FILE @ 18:17 ON 05/05/2021 - ALL FUTURE EDITS SHOULD BE MADE THERE.
# The Johnson-Lindenstrauss (JL) Lemma is the math behind Euclidean Distance/Space; it is what proves the “approximate maintenance of distance between the data points in different dimensions” property to be true. The lemma states that a small set of points in a high-dimensional space can be embedded into a space of much lower dimension in such a way that distances between the points are nearly preserved. The function `johnson_lindenstrauss_min_dim` of [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.johnson_lindenstrauss_min_dim.html)
# calculates the  minimum number of components *k*, i.e. the number of dimensions in which distances between 
# the points are nearly preserved, by the following formula:
# 
# $$
# k >= \frac{4 * log(n\_samples)} {(\frac{\epsilon^2}{2} - \frac{\epsilon^3}{3})}
# $$
# 
# Nevertheless, this only holds if the *right* *k* dimensions are chosen and not just any *k* dimensions.
# 
# The following outlines the key components of the JL Lemma and what they represent. 
# 
# - *k*: This represents the minimum number of dimensions to which the dataset can be reduced to without a substantive decrease in accuracy, down from the original *d* dimensions. This *k* is in effect the result obtained from the JL Lemma formula, based on the parameters provided thereto. 
# - $\epsilon$: This represents the error term. Namely, in the context of conducting random projections in Euclidean space it is the **approximate** distance that is maintained, i.e. there is some error involved during this process. Naturally, that error could result in either an increase or a decrease in the distance, which is portrayed by the inequality below. In this inequality, 1 represents the original distance, $\epsilon$ represents the error the user is willing to accept (range from 0 to 1, with lower values indicating a lower tolerance for error), the superscript indicates that this inequality applies to Euclidean Space (and hence the L2 norm), the vectors prior to their transformation are portrayed by the middle term ($f(x_i) - f(x_j)$), while the outer terms represent the two possibilities for the vectors following their transformation, namely that they are either somewhat smaller or somewhat larger than their original versions.
# $$
# (1 - \epsilon) ||x_i - x_j||^2 <= ||f(x_i) - f(x_j)||^2 <= (1 + \epsilon) ||x_i - x_j||^2
# $$
# - *n_samples*: This refers to the number of observations in the dataset.
# 
# **Note**: The number of dimensions is independent of the original number of features but instead depends on the size of the dataset: the larger the dataset, the higher is the minimal dimensionality of an eps-embedding.
# 
# **Note**: In the context of dimensionality reduction, random projections are typically used when one is unable to reliably calculate the covariance matrix (due to data sparsity for example), whereas when the covariance can be reliably calculated then Principal Component Analysis (PCA) is used. Both PCA and random projections require the dataset to be in Euclidean Space in order to function properly (more on Euclidean Space below).

# ## Introduction to the Datasets
# For this assignment, two datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php.) were chosen. Below is an overview of these datasets.  
# 
# **Classification of Dry Beans**  
# The dataset is used to classify dry beans into seven different classes; it is a multi-class classification. The dataset contains 13,611 observations and 17 features, one of which is the target-feature. It is an imbalanced dataset, as some classes are much more frequent than others. Thus, the performance of the classifier will be evaluated using the `f1_score` with the parameter settings `average = weighted`. The dataset is not yet standardized.  
# Data Source: https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset
# 
# 
# **Classification of Frogs:**  
# The dataset is used to classify the species of frogs. As the dataset encompasses ten different classes, i.e. species, it is a multi-class classification. The dataset contains 7,195 observations and 23 features, one of which is the target-feature (after deleting the columns *Family*, *Genus* and *RecordID*). As with the first dataset, the performance of the classifier will be evaluated using the `f1_score` with the parameter settings `average = weighted` as this dataset is again imbalanced. The dataset is already standardized.  
# Data Source: https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29

# In[ ]:




