import pandas as pd
from IPython.display import display, Markdown
import time
import importlib
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import datasets

import matplotlib.pyplot as plt
import numpy as np

class RandomSparseRepresentation:
    """This class executes the RandomSparseRepresentation"""
    def __init__(self):
        self._printmd("Welcome to the interface of **RandomSparseRepresentation**! :) \
\n\nYou have now instantiated an object, with which you can create a RandomSparseRepresentation. \
In order to do so, please first pick a dataset from this website [UCI ML](https://archive.ics.uci.edu/ml/index.php). \n\n\
Once you have done so, please use the function ```get_data()``` on your object to download that data. \
This function takes one necessary parameter and an optional one. The necessary one is the URL to \
the dataset you obtain when you right click in the data folder on the dataset and copy that link. \
Should the dataset not be a .csv within the datafolder on the UCI website, but rather a `.data` \
please also provide the column names as a list, which you can find in the `.names` file at UCI.")
        
    def get_data(self, url: str, names: list = None):
        try:
            if names:
                self.data = pd.read_csv(url, names = names)
            else:
                self.data = pd.read_csv(url)
            self._printmd("You successfully downloaded your dataset to the object! \n\n\
Now we can go ahead and split the data. \
Please call the `split_data()` function for it. You can pass it the `test_size` parameter, to split your \
data into test and train sets, the default value is `0.3`. Here are the first 5 rows of our data:")
            display(self.data.head(5))
        except BaseException as e:
            raise e
    
    def split_data(self, test_size = 0.3):
        self._printmd(f"The first thing we need to do, is to determine which of the columns shall be our target variable. \
Hence they are all printed out in the next step. \n\n\
{[x for x in self.data.columns]} \n\n\
In the next step please input a column name, which is contains your target variable.")
        time.sleep(1)
        target = input(prompt = "Please input your target variable here: ")
        self.X, self.y = self.data.drop(target, axis = 1), self.data[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size = test_size, random_state = 11)
        self._printmd(f"Your data has now be splitted into a train and test set by a ratio of `{test_size}`. \
This was done, by selecting the column `{target}` as target column and the rest as independent variables.")
    
    def JL_lemma(self, epsilon=0.1):
        self._printmd(f"In general, the theory of Professor Johnson and Professor Lindenstrauss posits \
the amount of columns to which we can reduce our dataset without losing any distance related information. \
We can specify a parameter called `epsilon` which determines the margin in which the distance is contained. \n\n\
Our current dataset has {self.data.shape[0]} observations. Using the JL algorithm, we could reduce it to \
{johnson_lindenstrauss_min_dim(self.data.shape[0], eps = epsilon)} dimensions.")
        if johnson_lindenstrauss_min_dim(self.data.shape[1], eps = epsilon) > self.data.shape[1]:
            self._printmd("The JL also works, if we have a smaller dataset... Ask group!")
        self._printmd("The next step is to set a define a baseline metric, on which we want to evaluate \
our algorithm with our later reduced dataset. For this please call the function `baseline()`.")
    
    def baseline(self, model = None):
        if not model:
            raise AttributeError("Please specify the model for your baseline metric! This can be done like \
`model = LinearSVC`, whereas LinearSVC refers to the function from sklearn.svm.")
        try:
            self.mod = model()
            self.mod.fit(self.X_train, self.y_train)
        except BaseException as e:
            raise e
        self._printmd("Next we need to choose a metric which should be used for our baseline. There are a few \
metrics we can choose from, the needed API (which you need to input next) can be viewed \
[here](https://scikit-learn.org/stable/modules/model_evaluation.html). Please choose your metric and input it \
in the following prompt. Be sure to insert it like `accuracy_score` if you want to use `accuracy_score` \
or respective for all other metrics.")
        self.metric = input(prompt = "Please insert your metric here: ")
        self.baseline = getattr(metrics, self.metric)(self.mod.predict(self.X_test), self.y_test)
        self._printmd("Awesome, you have set your baseline! Now call the function `apply_random_projection` to check out, \
how good your model performs when we reduce its dimensions.")
        
    def apply_random_projection(self):
        self._printmd("Now we can apply our random project onto our dataset, we loaded earlier. Once that function is done \
you can head over to the next function which is called `plot`. That function will plot the baseline and your \
chose metric over the different dimensions.")
        self.accuracies = []
        self.dims = np.int32(np.linspace(2, self.data.shape[1], 20))
        # Loop over the projection sizes, k
        for dim in self.dims:
            # Create random projection
            sp = SparseRandomProjection(n_components = dim)
            X = sp.fit_transform(self.X_train)

            # Train classifier of your choice on the sparse random projection
            model = self.mod
            model.fit(X, self.y_train)

            # Evaluate model and update accuracies
            test = sp.transform(self.X_test)
            self.accuracies.append(getattr(metrics, self.metric)(self.mod.predict(test), self.y_test))
            
    def plot(self):
        # Create figure
        plt.figure()
        plt.xlabel("# of dimensions k")
        plt.ylabel(f"{self.metric}")
        plt.xlim([2, self.data.shape[1]])
        plt.ylim([0, 1])

        # Plot baseline and random projection accuracies
        plt.plot(self.dims, [self.baseline] * len(self.accuracies), color = "r")
        plt.plot(self.dims, self.accuracies)

        plt.show()
        
    def _printmd(self, string):
        return display(Markdown(string))