import time
from sklearn import metrics
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

class Surrogate_data(object):
    """This class creates the surrogate data"""
    
    def __init__(self, n_samples = 1000, centers = 5,
                 cluster_std = 0.5, noise = 0.01,
                 n_classes = 5, n_informative = 10):
        self.n_samples = n_samples
        self.centers = centers
        self.cluster_std = cluster_std
        self.noise = noise
        self.n_classes = n_classes
        self.n_informative = n_informative
        
    def _create_surrogate_data(self):
        blob_X, blob_y = datasets.make_blobs(n_samples = self.n_samples, centers = self.centers,
                                             cluster_std = self.cluster_std, random_state = 1)
        circles_X, circles_y = datasets.make_circles(n_samples = self.n_samples, noise = self.noise,
                                                     random_state = 1)
        class_X, class_y = datasets.make_classification(n_samples = self.n_samples, n_classes = self.n_classes,
                                                        n_informative = self.n_informative, random_state = 1)
        hastie_X, hastie_y = datasets.make_hastie_10_2(n_samples = self.n_samples, random_state = 1)
        moons_X, moons_y = datasets.make_moons(n_samples = self.n_samples, noise = self.noise, random_state = 1)
    
        self.datasets = [
            (blob_X, blob_y, "make_blobs"),
            (circles_X, circles_y, "make_circles"),
            (class_X, class_y, "make_classification"),
            (hastie_X, hastie_y, "make_hastie"),
            (moons_X, moons_y, "make_moons")
        ]
        
    def _fit_predict(self, X, y, classifier, **kwargs):
        # fit model
        if classifier.__name__ in ["GaussianMixture", "KMeans"]:
            n_components = len(np.unique(y))
            try:
                model = classifier(n_components = n_components, **kwargs)
            except TypeError:
                model = classifier(n_clusters = n_components, **kwargs)
        else:
            model = classifier(**kwargs)
        t0 = time.time()
        model.fit(X)
        t1 = time.time()
        # predict latent values
        if classifier.__name__ in ["DBSCAN", "HDBSCAN", "AgglomerativeClustering"]:
            return model.fit_predict(X), ('%.2fs' % (t1 - t0)).lstrip('0')
        return model.predict(X), ('%.2fs' % (t1 - t0)).lstrip('0')
    
    def plot_raw_vs_predict(self, classifier, **kwargs):
        # creating surrogate data
        self._create_surrogate_data()
        # For plotting in subplots later
        fig, ax = plt.subplots(len(self.datasets), 2, figsize = (14,14))
        fig.suptitle(f"{classifier.__name__} Comparision between raw and clustered data", ha = "center", va = "top", x = 0.5, y = 1.05, fontsize = 16)
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=1, wspace=.15,
                    hspace=.25)
        # Create the actual plots
        for plot_num, (dataset_X, dataset_y, name) in enumerate(self.datasets):
            yhat, time = self._fit_predict(dataset_X, dataset_y, classifier, **kwargs)

            ax[plot_num, 0].scatter(dataset_X[:,0], dataset_X[:,1], c = dataset_y)
            ax[plot_num, 0].set_title(f"{name} function")
            ax[plot_num, 1].scatter(dataset_X[:,0], dataset_X[:,1], c = yhat)
            ax[plot_num, 1].set_title(f"{name} function")
            ax[plot_num, 1].text(.99, .01, time,
                 transform=ax[plot_num, 1].transAxes, size=15,
                 horizontalalignment='right')
            try:
                ax[plot_num, 1].text(.01, .01, round(metrics.silhouette_score(dataset_X, yhat),2),
                 transform=ax[plot_num, 1].transAxes, size=15,
                 horizontalalignment='left')
            except ValueError:
                ax[plot_num, 1].text(.01, .01, "None",
                 transform=ax[plot_num, 1].transAxes, size=15,
                 horizontalalignment='left')
        plt.show()