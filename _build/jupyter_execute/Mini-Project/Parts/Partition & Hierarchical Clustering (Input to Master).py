#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# # Partition Clustering
# A key characteristic of Partition Clustering algorithms is that they require the user to specify the number of clusters which the algorithm will find. Possessing at least some degree of domain knowledge and/or insight into the dataset is quite helpful in this regard, as otherwise the number of clusters formed is arbitrary and hence is not likely to reflect the inherent number of clusters within the data. On the other hand, the fact the programmer specifies the number of clusters to be identified does help Partition Clustering algorithms to be relatively efficient (computationally) when compared with other clustering algorithms. Partition Clustering algorithms cluster all data points, regardless if a given data point could be reliably said to be part of a cluster.

# ## K-means
# ### General Description & Application
# K-means is a very popular Partition Clustering algorithm. Essentially, the user specifies the number of clusters to be identified, and the algorithm iteratively adjusts the clusters by moving what are known as **Centroids** in a manner that minimizes the distance of the data points to the Centroid to which they are assigned. In so doing the specific Centroid to which a given datapoint is assigned can change, as the datapoints are assigned to the nearest Centroid and as mentioned the Centroids iteratively change locations accordingly.    
# 
# The major benefit of K-means is its minimal computational cost - it is a relatively simple and efficient algorithm that is well-suited to working with large datasets. However, as discussed in the description of Partition Clustering algorithms more broadly this can also be a downside, especially without domain knowledge and/or insight into the dataset. Furthermore, as with other Partition Clustering algorithms the K-means algorithm will assign all points to a cluster, irrespective of whether a given point is actually part of a cluster. 
# 
# ### Steps
# The K-means algorithm can be broken down into four specific steps: 
# 1. Determine K, the number of clusters to be identified.
# 2. Select K data points to serve as the initial centroids.
# 3. Assign each data point to the closest centroid.
# 4. Move the centroids according to the new "average location" of the data points assigned to each centroid. $\newline$
# 
# $\underline{Note}$: Steps 3 and 4 are repeated until there are no further changes to the clusters to which each data point is assigned or once the `max_iter` parameter has been reached. In order try and find a global rather than a local maximum, the algorithm is initialized with different centroid seeds `n_init` times, meaning that the total number of times the algorithms is effectuated is the product of the `n_init` and `max_iter` parameters.
# 
# 
# ### Select Parameters (Scikit Learn Implementation)
# `n_clusters`: The number of clusters to form, which is also the number of centroids to generate.
# $\newline$
# `n_init`: Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
# $\newline$
# `max_iter`: Maximum number of iterations of the k-means algorithm for a single run.
# $\newline$
# **Note**: Parameter names and descriptions were obtained from the official Scikit Learn documentation (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).

# ## Spectral Clustering
# ### General Description & Application
# 
# The idea of spectral clustering is rooted in graph theory. The spectral clustering algorithm aims to identify communities of nodes in a graph based on connections between them. It can be understood as aiming to maximize the number of within-cluster connections and to minimize the number of between-cluster connections. The spectral clustering algorithm also allows for clustering of non graph data. Thus, points that are (immediately) next to each other, i.e. closely connected, are identified in dataset.
# 
# The spectral clustering algorithm utilizes information from the eigenvalues and eigenvectors of the Laplacian Matrix. The calculation of the (unnormalized) Laplacian Matrix will be explained in more detail in a bit. In the end, a standard clustering algorithm, such as KMeans, is applied on the relevant eigenvectors of the Laplacian Matrix to identify clusters in the dataset.
# 
# We will now briefly outline some advantages and disadvantages of the spectral clustering algorithm. The spectral clustering algorithm is generally able to capture complex data structures as it does not make an assumption about the shape/form of the clusters. Nevertheless, the spectral clustering algorithm still requires us to specify the number of clusters beforehand as indicated by the `n_cluster` hyperparamter. In that sense, it has the same disadvantage as K-Means. Furthermore, the spectral clustering algorithm groups every individual data point to a cluster, which means it may also cluster noise. Additionally, it is computationally expensive for large datasets.
# 
# ### Steps
# The general process of the spectral clustering algorithm implemented in [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html) with the function `sklearn.cluster.SpectralClustering` can be illustrated by the following steps:
# 1. Construct the Affinity Matrix based on the datapoints
# 2. Create the Degree Matrix based on the Affinity Matrix
# 3. Construct the Laplacian Matrix by subtracting the Affinity matrix from the Degree Matrix
# 4. Eigendecomposition of the Laplacian Matrix
# 5. Apply a standard clustering algorithm, e.g. KMeans, on the relevant eigenvectors of the Laplacian Matrix
# 
# The previously outlined steps will now be described in more detail: 
# 
# #### Step 1: Affinity Matrix 
# The entries of an Affinity Matrix show how similar points are to each other. The higher the entry in a Affinity Matrix, the higher the similarity between the points. The Affinity Matrix can be constructed in different ways. Therefore, Scikit-Learn's [implementation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering)  includes the parameter *affinity*, which defines how to construct the Affinity Matrix. Four options are available: `nearest_neighbors`, `rbf`, `precomputed`and `precomputed_nearest_neighbors`.
# - `nearest_neighbors` constructs the affinity matrix by computing a graph of nearest neighbors. If chosen, the hyperparameter `n_neighbors` also needs to be set as this determines the number of neighbors to use when constructing the affinity matrix.
# - `rbf` constructs the affinity matrix using a radial basis function (RBF) kernel
# - `precomputed` interprets X as a precomputed affinity matrix, where larger values indicate greater similarity between instances
# - `precomputed_nearest_neighbors` interprets X as a sparse graph of precomputed distances, and construct a binary affinity matrix from the n_neighbors nearest neighbors of each instance
# 
# It will now be detailed, how the Affinity Matrix is constructed using `nearest_neighbors` and `rbf`, i.e. in those cases where no precomputed Affinity Matrix is provided:
# 
# **nearest_neighbors:**  
# When setting the parameter *affinity* to `nearest_neighbors`, the Affinity Matrix is calculated using the k-nearest neighbors method. Thus, the number of neighbors to use when constructing the Affinity Matrix needs to be specified with the parameter `n_neighbors`. Let's call the matrix, which stores the relationships of k-nearest neighbours, *Connectivity Matrix*. If another datapoint belongs to the k-nearest neighbors, the Connectivity Matrix will indicate it with an entry of 1. If it does not belong to the k-nearest neighbors, it will be indicated with a 0. In Scikit-Learn, the Affinity Matrix is then calculated using the following [formula](https://github.com/scikit-learn/scikit-learn/blob/aa898de885ed4861a03e4f79b28f92f70914643d/sklearn/cluster/_spectral.py#L512):
# 
# $$
# {Affinity\ Matrix} = 0.5 * ({Connectivity\ Matrix} + {Connectivity\ Matrix^T})
# $$ 
# 
# Thus, each entry in the Affinity Matrix can only take up one of three possible entries: 1.0, 0.5 or 0.0. 
# - `1.0` indicates that when calculating the closest k-nearest neighbors, both datapoints were amongst the respective nearest k-datapoints 
# - `0.5` indicates that this was only true for one datapoint, i.e. only in one "direction"
# - `0.0` indicates that for both datapoints, the other respective datapoint was not among the k-nearest
# 
# Let's assume the following example with n_neighbors = 2:
# 
# <div>
# <img src="Spectral_Clustering/Affinity_Matrix_nearest_neighbor.png" width="700"/>
# </div>
# 
# For data point 0, the k-nearest neighbors (including itself) are data point 0 and data point 1. For data point 1, only data point 1 and data point 2 are the k-nearest neighbors. In turn, the Affinity Matrix shows 0.5 at the entry row 0, column 1. 
# *Note*: The parameter `include_self` is set to *True*, which means that each sample is marked as the first nearest neighbor to [itself](https://github.com/scikit-learn/scikit-learn/blob/aa898de885ed4861a03e4f79b28f92f70914643d/sklearn/cluster/_spectral.py#L510).
# 
# **rbf:**  
# Setting the parameter *affinity* to `rbf`, i.e. also its [default setting](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html), the Affinity Matrix is constructed using a kernel function with Euclidean distance d(X, X), i.e. it calculates the pairwise distances of all points in our dataset *X*: 
# 
# `np.exp(-gamma * d(X,X) ** 2)`
# 
# The default for `gamma` is 1.0. Here, the entries of the Affinity Matrix can take any value between 0.0 and 1.0, where an increase in value corresponds to an increase in similarity.
# 
# <div>
# <img src="Spectral_Clustering/Affinity_Matrix_rbf.png" width="700"/>
# </div>
# 
# #### Step 2: Degree Matrix 
# The Degree Matrix is a diagonal matrix, which is obtained by taking the sum of each row in the Affinity Matrix. The entries on the diagonal are called *degree*. Thus, the closer the individual points are group together, the higher the entry in the diagonal as each individual entry in the rows of the Affinity Matrix will be larger. At the same time, if the points are spread out further, the individual entries in the rows of the Affinity Matrix are smaller, which in turn leads to a smaller sum of the row.
# 
# #### Step 3: Laplacian Matrix
# The (unnormalized) Laplacian Matrix is calculated by subtracting the Affinity Matrix from the Degree Matrix.
# 
# $$
# L = D - A
# $$
# 
# The (unnormalized) Laplacian Matrix has among others the following [basic properties](http://www.tml.cs.uni-tuebingen.de/team/luxburg/publications/Luxburg07_tutorial.pdf):
# - L is symmetric and positive semi-definite
# - The smallest eigenvalue of L is 0, the corresponding eigenvector is the constant one vector 
# - L has n non-negative, real-valued eigenvalues 0 = $??_1$ ??? $??_2$ ??? . . . ??? $??_n$
# 
# *Note:* There are several other variants of the Laplacian Matrix as well as spectral clustering algorithms, which were out of the scope of this assignment as we did not solely focus on this type of algorithm.
# 
# In the following, the Affinity Matrix, Degree Matrix and Laplacian Matrix are illustrated for the example using `rbf` for the parameter *affinity*.
# 
# <div>
# <img src="Spectral_Clustering/Laplacian_Matrix_rbf.png" width="700"/>
# </div>
# 
# #### Step 4: Eigendecomposition of Laplacian Matrix
# In the next step, the eigenvalues and eigenvectors of the Laplacian Matrix are calculated. As already outlined in our introduction to *Partition Clustering*, determining the number clusters in a dataset is generally a difficult task. Similar to the *elbow plot*, which may be used when determining the "right" number of clusters for the KMeans algorithm, we can make use of the eigengap heuristic in spectral clustering. The goal is to identify the first large gap between the eigenvalues, which are ordered increasingly. Thus, we choose the number of clusters such that all eigenvalues $??_1$,..., $??_k$ are minuscule and $??_{k+1}$ is comparatively large. The number of eigenvalues $??_1$,..., $??_k$ before this gap generally correspond to the number of clusters in our dataset.
# 
# As can be seen by the following stylized examples, we can identify a gap between eigenvalues number two and three as well as between number four and five, which in turn helps to determine the "right" number of clusters. 
# 
# <div>
# <img src="Spectral_Clustering/Eigenvalues_Laplacian.png" width="700"/>
# </div>
# 
# *Note:*
# Thoses examples are just for illustration of the eigengap heuristic. Here, kmeans could easily be applied and would yield a good and fast solution. Furthermore, it has to be noted that the parameter `n_cluster` needs to be specified before the spectral clustering algorithm is run. Thus, the visualization of the eigengap heuristic is just used to illustrate the information content of the eigenvalues of the Laplacian Matrix. It should not be understood as a step, which can be performed when calling `sklearn.cluster.SpectralClustering`.
# 
# #### Step 5: Application of standard clustering algorithm 
# The k eigenvectors associated with the k smallest eigenvalues are used for the partitioning of the dataset, except for the eigenvector corresponding to the first eigenvalue as this eigenvalue will always have a value of zero (see properties further above). A standard clustering algorithm is applied on the set of eigenvectors. The default clustering algorithm is KMeans.
# 
# ### Select Parameters (Scikit Learn Implementation)
# **n_cluster:** The parameter *n_clusters* defines the dimension of the projection subspace, i.e. the number of clusters.
# 
# **eigen_solver:**
# This parameter determines which eigenvalue decomposition strategy to use.
# 
# **affinity:** 
# This parameter defines how to construct the affinity matrix. Four options are available: `nearest_neighbors`, `rbf`, `precomputed`and `precomputed_nearest_neighbors` (see above).
# 
# **n_init:**  
# Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia. Only used if assign_labels='kmeans'.
# 
# **assign_labels:**  
# The strategy for assigning labels in the embedding space, i.e. the clustering algorithm. There are two ways to assign labels after the Laplacian embedding: `kmeans` and `discretize`. k-means is a popular choice, but it can be sensitive to initialization. Discretization is another approach which is less sensitive to random initialization`
# 
# **Note**: Parameter names and descriptions were obtained from the official Scikit Learn [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering).
# 
# **Sources:**  
# - Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow by Aurelien Geron (2020)
# - https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html  
# - https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering
# - https://towardsdatascience.com/spectral-clustering-aba2640c0d5b
# - http://www.tml.cs.uni-tuebingen.de/team/luxburg/publications/Luxburg07_tutorial.pdf
# - https://towardsdatascience.com/spectral-clustering-82d3cff3d3b7
# - https://www.mygreatlearning.com/blog/introduction-to-spectral-clustering/

# # Hierarchical Clustering
# Hierarchical Clustering algorithms create a hierarchy of clusters using a predefined distance metric such as Single-Link (comparing the observation to the closest point in each cluster) vs. Complete-Link (comparing the observations to the farthest point in each cluster). Hierarchical Clustering algorithms will stop running when either (1) the specified number of clusters has been obtained, or (2) the linkage distance threshold has been reached. As with Partition Clustering, Hierarchical Clustering algorithms cluster all observations in the dataset.
# 
# At a high level, we can think of an entire dataset as being a single cluster, irrespective of how dispersely the datapoints contained therein are distributed. Along a similar train of thought, the most granular way to cluster a dataset would be to assign each datapoint to its own cluster; in the context of Hierarchical Clustering these are known as **singletons**. So, with hierarchical clustering algorithms what we have is a way to outline the different ways in which a given dataset can be clustered, ranging from a single cluster that contains the entire dataset to *n* clusters where *n* is equal to the number of datapoints. This range or "hierarchy" of clusters can be thought of as representing different degrees of granularity in terms of the similarity between the data points, where the singletons are the most granular groupings as each data point has its own cluster.
# 
# Hierarchical clustering algorithms can be further categorized based on whether a **top-down** or a **bottom-up** approach is used to cluster the data. With a top-down approach, the algorithm starts with the dataset as a whole (one cluster) and iteratively breaks it down into increasingly smaller clusters. Conversely, with a bottom-up approach the algorithm starts with the singletons as individual clusters (that is the initialization step) and iteratively combines them into ever-larger clusters. As mentioned, the deciding factor in terms of how the hierarchy of the clusters is formed is the stipulated distance metric, irrespective of whether a top-down or a  bottom-up approach is followed.
# 
# A common and very useful visual representation of how hierarchical clustering algorithms work is known as a "dendrogram", which our Python implementation provides for each of the two hierarchical clustering algorithms we researched. Essentially, this can be thought of as a hierarchical tree of clusters, with bottom row representing the singletons which progressively weave together until they are all attached via the uppermost node. 
# 
# Our discussion of Hierarchical Clustering algorithms focuses on two specific algorithms: (1) Agglomerative Clustering and (2) Birch Clustering.

# ## Agglomerative Clustering
# ### General Description & Application
# 
# Agglomerative Clustering employs a bottom-up approach; the algorithm starts with the individual singletons and iteratively combines them into ever-larger clusters until either (1) the specified *n_clusters* parameter is reached, or (2) the specified distance threshold is reached. If the distance threshold is too large than a single "cluster" that contains the entire dataset will be returned. With Agglomerative Clustering each iteration reduces the number of clusters by one.
# 
# Relative to top-down hierarchical clustering algorithms Agglomerative Clustering is much less efficient computationally. That said, the greater computational burden of this algorithm does help to ensure nearby points are assigned to the appropriate cluster. 
# 
# ### Steps
# The Agglomerative Clustering algorithm can be broken down into three distinct steps:
# 1. Initialize *n* singleton clusters, where *n* is the number of datapoints.
# 2. Determine those two clusters that are closest together, based on the specified distance metric.
# 3. Merge the two clusters identified in Step 2 into a single cluster.
# 
# $\underline{Note}$: Steps 2 and 3 are repeated until either (1) the specified number of clusters has been obtained, or (2) the linkage distance threshold has been reached.
# 
# ### Select Parameters (Scikit Learn Implementation)
# `n_clusters`: The number of clusters to find. It must be ``None`` if ``distance_threshold`` is not ``None``.
# $\newline$
# `affinity`: Metric used to compute the linkage. Can be ???euclidean???, ???l1???, ???l2???, ???manhattan???, ???cosine???, or ???precomputed???. If linkage is ???ward???, only ???euclidean??? is accepted. If ???precomputed???, a distance matrix (instead of a similarity matrix) is needed as input for the fit method.
# $\newline$
# `linkage`: Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.
# 
# - ???ward??? minimizes the variance of the clusters being merged.
# 
# - ???average??? uses the average of the distances of each observation of the two sets.
# 
# - ???complete??? or ???maximum??? linkage uses the maximum distances between all observations of the two sets.
# 
# - ???single??? uses the minimum of the distances between all observations of the two sets.
# 
# `distance_threshold`: The linkage distance threshold above which, clusters will not be merged. If not ``None``, ``n_clusters`` must be ``None`` and ``compute_full_tree`` must be ``True``.
# $\newline$
# **Note**: Parameter names and descriptions were obtained from the official Scikit Learn documentation (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html).
# $\newline$
# **Dendogram Implementation**: 'https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py`

# ## BIRCH
# ### General Description & Application
# 
# BIRCH is an acronym for **B**alanced **I**terative **R**educing and **C**lustering using **H**ierarchies. The algorithm was introduced in 1996 by Tian Zhang, Raghu Ramakrishnan and Miron Livny in their [article](https://www2.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf) *BIRCH: An Efficient Data Clustering Method for Very Large Databases*. 
# 
# As can already be inferred from the title, BIRCH is designed for clustering of very large datasets. BIRCH takes into account that the amount of memory is generally limited, i.e. the size of the dataset generally exceeds the available memory. Thus, BIRCH aims to minimize I/O costs as it does not require to memorize the entire data set. The algorithm can yield a satisfactory clustering of the data set with just a single scan of the data set. To increase the performance, just a few additional scans of the data set are needed.
# 
# BIRCH belongs to the category of Hierarchical Clustering algorithms. In contrast to Agglomerative Clustering, it uses a top-down-approach instead of a bottom-up-approach. The *Clustering-Feature (CF)* and the *CF-Tree* are two key concepts of the algorithm. Based on the data set, BIRCH constructs a *Clustering Feature Tree (CF-Tree)*. The CF-Tree consists of *Clustering Feature nodes (CF Nodes)*, which in turn contain *Clustering Features*. Clustering Features basically summarize relevant statistical metrics for the given cluster. Thus, BIRCH allows for clustering of larger datasets by first generating a compact summary of the large dataset that preserves as much information as possible. Secondy, this summary is then clustered instead of clustering the entire, i.e. larger, dataset. Both, *CF-Trees* and *Clustering Features* will be explained in more detail below.
# 
# In general, the main advantage of BIRCH is its scalability as it yields satisfactory results in a comparatively small amount of time for very large datasets. Its time complexity is $=O(n)$, where n
# equals the number of objects to be clustered. One disadvantage of BIRCH is that it can only metric attributes, i.e. no categorical variables. Moreover, if the shape of the clusters are not spherical, the algorithm may not perform well as it utilizes the radius to control the boundary of a cluster.
# 
# #### CF-Tree
# The [CF-Tree](https://www2.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf) is a "height-balanced tree with two parameters: branching factor B and threshold T." The following [picture](http://avid.cs.umass.edu/courses/745/f2010/notes/DataMining.htm) depicts an example of the structure of a CF-Tree:
# 
# <div>
# <img src="CF_Tree_Structure.png" width="500"/>
# </div>
# 
# `Question to Group: B = 7 does not really make sense to me. Shouldn't it be B=6?`
# 
# Naturally, internal nodes or non-leaf nodes of the CF-tree have descendants or "children". They take the form $[CF_i, child_i]$, where $i = 1, 2, ..., B$. Thus, each non-leaf node contains at maximum *B* entries, where *B* represents the branching factor and each entry one associated subcluster. In turn, $B$ also affects the size of the CF-Tree. The larger *B*, the smaller the CF-Tree. In `sklearn.cluster.Birch`, the branching factor *B* is represented by the hyperparameter `branching_factor`. The Clustering Features, $CF_i$, store the information about the descendants, i.e. the subclusters, while $child_i$ is used as a pointer to the i-th child/descendant. Thus, the Clustering Feature, $CF_i$, contains the sums of all Clustering Features of $child_i$.  
# 
# The leaf nodes are of the form $[CF_i]$. Furthermore, each leaf node has two entries "prev" and "next", which are used to connect all leaf nodes. This chain allows for efficient scans. A leaf node contains at most *L* entries. While each entry in leaf nodes also represents an associated subcluster, i.e. equivalent to non-leaf nodes, the entries have additionally to comply to the treshold requirement *T*. The radius of the subcluster represented by each entry has to be smaller than *T*. In turn, the threshold *T* also affects the size of the CF-Tree. With a smaller threshold, the size of the CF-tree will increase and with a larger threshold, the size of the CF-tree will decrease. In `sklearn.cluster.Birch`, the threshold *T* can be adjusted with the hyperparameter `threshold`. 
# 
# #### Clustering Feature
# 
# The [Clustering Feature](https://www2.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf) is defined as "a triple summarizing the information that we maintain about a cluster": 
# 
# $$
# CF = (N,{LS},SS)
# $$
# 
# - $N$: number of data points in the cluster
# - ${LS}$: linear sum of the N data points, i.e. $\sum_{i=1}^N {X_i}$
# - $SS$: square sum of the N data points, i.e. $\sum_{i=1}^N ({X_i})^2$
# 
# Suppose there are three points, (1,4), (2,2) and (4,3) in a (sub-)cluster $C_i$.   
# The Clustering Feature $CF_i$ = $(3, (1+2+4, 4+2+3), (1^2+2^2+4^2, 4^2+2^2+3^2)) = (3, (7, 9), (21, 29))$.
# 
# A Clustering Feature can be understood as a condensed summary of data points, which captures the natural closeness of the data. Thus, it is much more efficient as it does not require to store all the data points. Moreover, the Clustering Feature allows to derive many other useful statistics of a cluster such as the centroid $C$ or radius *R*, where *R* represents the average distance from the data points to the centroid.
# 
# $
# C = \frac{\sum_{i=1}^N  x_i}{N} = \frac{LS}{N}
# $
# 
# $
# R = \sqrt{\frac{ \sum_{i=1}^N ({X_i} - {C})^2}{N}} = \sqrt{\frac{SS}{N} - (\frac{{LS}}{N})^2}
# $
# 
# Other useful metrics that can be calculated, are among others the *Centroid Euclidan Distance*, the *Manhattan Distance* or *Average Inter-Cluster Distance*. 
# 
# As outlined earlier, the Clustering Feature $CF_i$ at an internal node contains the sum of all Clustering Features of its descendants. That is because Clustering Features are additive. Thus, when two clusters $C_2$ and $C_3$ with the Clustering Features $CF_2$ = $(N_2,LS_2,SS_2)$ and $CF_3$ = $(N_3,LS_3,SS_3)$ are merged, then the resulting cluster $C_1$ simply consists of $CF_2$ + $CF_3$ = $(N_2+N_3,LS_2+LS_3,SS_2+SS_3)$.
# 
# #### Insertion Algorithm
# The CF-Tree is built dynamically as new data points are added. Thus, the CF-Tree directs a new insertion into the correct subcluster similar to a B+-Tree, which sorts new data points into their correct position. In the following, the general steps of the insertion of an entry (data point or subcluster) into the CF-Tree are described:<br>
# **1. Identifying the appropriate leaf:**<br>
# Starting from the top of the CF-Tree, i.e. the root, the algorithm recursively descends down the CF-Tree to find the closest child node based on a certain distance metric. In `sklearn.cluster.Birch`, Euclidean Distance is [used](https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/cluster/_birch.py#L73).<br>
# 
# **2. Modifying the leaf:**<br>
# Once the leaf node, e.g. $CF_8$, is reached, the closest leaf-entry, e.g. $CF_{96}$, in that node is found. Afterwards, it is checked if $CF_{96}$ can absord the new entry without violating the threshold requirement *T*. If it can, i.e. the radius of $CF_{96}$ remains smaller than *T* even after adding the new entry, $CF_{96}$ will be updated. If adding the entry would lead to a violation of the threshold requirement *T*, a new entry for a Clustering Feature, i.e. $CF_i$, will be added to the leaf. This can only be done if there is space for another entry on the leaf, i.e. the number of Clustering Features on that leaf is smaller than *L*. Otherwise, the leaf node is splitted. The node is splitted by choosing the pair of entries which are the farthest apart as seeds. All other entries are then redistribution to the closest one.
# 
# **3. Modifying the path to the leaf:**<br>
# As we previously outlined, every internal, i.e. non-leaf, node is composed of the Clustering Features of all its descendants. Thus, upon inserting an entry into a leaf node, the information for each internal node on the path towards the leaf node needs to be updated. If the leaf-node was splitted in the previous step, a new non-leaf entry is inserted into the parent node. This newly inserted entry at the parent node will point to the newly created leaf in the previous step. Here, the branching factor *B* must be adherred to. If the parent node does not have enough space as it already contains *B* entries, the parent node must be split as well. This splitting is performed up to the root.  
# 
# `Question to Group: Should I try to visualize the different possibilities?`
# 
# In general, if the size required for storing the CF-Tree still exceeds the size of the memory, a larger theshold value can be specified to rebuild a smaller CF-Tree, which fits into the memory.
# 
# ### Steps
# In general, BIRCH utilizes a multiphase clustering technique consisting of [four phases](https://www2.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf).
# 
# <div>
# <img src="Phases_BIRCH.png" width="400"/>
# </div>
# 
# Two out of the four phases are optional as a single scan of the data set already yields a good CF-Tree structure. In the following, the two obligatory phases are illustrated. <br>
# 
# **Phase 1: Construct the CF-Tree** <br> 
# BIRCH scans the data set to construct an initial CF-tree, which is stored in-memory. <br>
# 
# **Phase 3: Clustering** <br>
# After the CF-Tree is built, any clustering algorithm can be used to cluster the leaf nodes of the CF-Tree. In `sklearn.cluster.Birch`, by default [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering) is used. This is determined by the default-value of the hyperparameter `n_clusters`.
# 
# ### Select Parameters (Scikit Learn Implementation)
# **threshold:**<br>
# The radius of the subcluster obtained by merging a new sample and the closest subcluster should be lesser than the threshold. Otherwise a new subcluster is started. Setting this value to be very low promotes splitting and vice-versa.<br>
# 
# **branching_factor:**<br>
# Maximum number of CF subclusters in each node. If a new samples enters such that the number of subclusters exceed the branching_factor then that node is split into two nodes with the subclusters redistributed in each. The parent subcluster of that node is removed and two new subclusters are added as parents of the 2 split nodes.<br>
# 
# **n_clusters:**<br>
# Number of clusters after the final clustering step, which treats the subclusters from the leaves as new samples.
# 
# - None: the final clustering step is not performed and the subclusters are returned as they are.
# - sklearn.cluster Estimator : If a model is provided, the model is fit treating the subclusters as new samples and the initial data is mapped to the label of the closest subcluster.
# - int: the model fit is [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering) with n_clusters set to be equal to the int
# 
# **Sources:**
# - Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow by Aurelien Geron (2020)
# - Data Mining - Concepts & Techniques by Jiawei Han, Micheline Kamber, Jian Pei (2012)
# - https://www2.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf
# - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html
# - https://towardsdatascience.com/machine-learning-birch-clustering-algorithm-clearly-explained-fb9838cbeed9
# - https://scikit-learn.org/stable/modules/clustering.html#birch

# ## Notes to Self
# - $\underline{Remember}$: To enter math mode two dollar signs are required. 
# - Alternatively, writing "%%latex" can make the entire cell be in math mode.
# - Just like there is hard and soft classification, there is also hard and soft clustering. Soft clustering is where a given point is both in Cluster A and Cluster B, likely in different (i.e. weighted proportions). One type of well-known soft clustering algorithm are Gaussian Mixture Models. 

# ## Works Cited
# [1] https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html $\newline$
