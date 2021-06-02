from scipy.cluster.hierarchy import dendrogram
import numpy as np
import matplotlib.pyplot as plt

def plot_dendrogram(model, datasets,  **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    plt.figure(figsize=(16, 6))
    dendrogram(linkage_matrix, **kwargs)
    plt.title(f'Hierarchical Clustering Dendrogram for {datasets} data', fontdict = {'fontsize': 14})
    plt.xlabel("Number of points in node (or index of point if no parenthesis).", fontdict = {'fontsize': 14})
    plt.show()