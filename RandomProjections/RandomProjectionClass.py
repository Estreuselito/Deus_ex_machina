import pandas as pd
from IPython.display import display, Markdown, clear_output
import time
import importlib
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import datasets

import matplotlib.pyplot as plt
import numpy as np

class RandomSparseRepresentation:
    """This class executes the RandomSparseRepresentation"""
    def __init__(self, birthday_version = False, text: bool = True):
        if birthday_version:
            self._jans_birthday()
            self._printmd("---")
        if text:
            self._printmd(
"""
# Welcome to the interface of **RandomSparseRepresentation**!
        
You have now instantiated an object, with which you can create a RandomSparseRepresentation.
The function ```get_data()``` retrieves the data, which was previously downloaded from the sources outlined above.""")
        
    def get_data(self, url: str, data_type: str = None, names: list = None, text: bool = True, **kwargs):
        try:
            if names and data_type in [".csv", ".data"]:
                self.data = pd.read_csv(url, names = names, **kwargs)
            elif data_type in [".xls", ".xlsx"]:
                self.data = pd.read_excel(url, header = 0)
                self.data = self.data.dropna()
            elif url and data_type in [".csv", ".data"]:
                self.data = pd.read_csv(url)
            else:
                raise AttributeError(f"""The chosen `data_type = {data_type}` is currently not supported
                                     by this function. All supported `data_types` are `.csv`, `.data`, 
                                     `.xls`, `.xlsx` and everything else which can be read with `pd.read_csv
                                     or `pd.read_excel`.""")
            if text:
                self._printmd(
"""You successfully loaded your dataset to the object!

Now we can go ahead and split the data.
Please call the `split_data()` function for it. You can pass it the `test_size` parameter, to split your
data into test and train sets, the default value is `0.3`. Here are the first 5 rows of our data:""")
                display(self.data.head(5))
        except BaseException as e:
            print(e)
    
    def split_data(self, test_size = 0.3, standardize = True, columns_to_drop: list = None, text: bool = True):
        if text:
            self._printmd(
f"""The first thing we need to do, is to determine which of the columns shall be our target variable.
Hence they are all printed out in the next step.

{[x for x in self.data.columns]}

In the next step please input a column name, which contains your target variable.""")
            time.sleep(1)
        elif not text:
            self._printmd(f"The columns of this dataset are: {[x for x in self.data.columns]}")
        while True:
            target = input(prompt = "Please input your target variable here: ")
            if target not in self.data.columns:
                self._printmd(f"`{target}` could not be found in axis, please review your target choice")
                continue
            self.X, self.y = self.data.drop(target, axis = 1), self.data[target]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                        test_size = test_size,
                                                                                        random_state = 0)
            if standardize:
                # Standardize the Data
                sc = StandardScaler()
                self.X_train = sc.fit_transform(self.X_train)
                self.X_test = sc.transform(self.X_test)
            if columns_to_drop:
                self.X_train = self.X_train.drop(columns_to_drop, axis = 1)
                self.X_test = self.X_test.drop(columns_to_drop, axis = 1)                
            break
            
        if text:
            self._printmd(
f"""Your data has now be split into a train and test set by a factor of `{test_size}`.
This was done by selecting the column `{target}` as the target column and the rest as independent variables.""")
    
    def JL_lemma(self, epsilon=0.1):        
        self._printmd(
f"""In general, the theory of Professor Johnson and Professor Lindenstrauss posits
the amount of columns to which we can reduce our dataset without losing any distance related information.
We can specify a parameter called `epsilon` which determines the margin in which the distance is contained.

Our current dataset has {self.data.shape[0]} observations and {self.data.shape[1]} dimensions. Using the JL algorithm $k$ = 
{johnson_lindenstrauss_min_dim(self.data.shape[0], eps = epsilon)} dimensions.""")
        if johnson_lindenstrauss_min_dim(self.data.shape[1], eps = epsilon) > self.data.shape[1]:
            self._printmd(
"""Nevertheless, dimensionality reduction can still work with lower dimensions.""")
        self._printmd(
"""The next step is to set a baseline metric, which can be used to evaluate the algorithm with
the reduced dataset. For this please call the function `baseline()`.""")
        
    def baseline(self, model = None, text: bool = True, **kwargs):
        if not model:
            raise AttributeError("Please specify the model for your baseline metric! This can be done like \
`model = LinearSVC`, whereas LinearSVC refers to the function from sklearn.svm.")
        try:
            self.mod = model(**kwargs)
            self.mod.fit(self.X_train, self.y_train)
        except BaseException as e:
            raise e
        if text:
            self._printmd(
r"""In order to assess the performance of a classifier, it is important to incorporate a numerical evaluation of the algorithm. 
For this, a variety of performance measures are available. It is essential to make use of an adequate performance measure as 
their applicability and significance depend on the dataset as well as the specific classification task.
There are a few metrics we can choose from, the needed API (which you need to input next) can be viewed
[here](https://scikit-learn.org/stable/modules/model_evaluation.html). For the task at hand, the performance 
measures used are either *accuracy* or the $f_1$ *score*.

\begin{equation}
Accuracy = \frac{True\ Positives + True\ Negatives }{True\ Positives + False\ Positives + True\ Negatives + False\ Negatives}
\end{equation}

*Accuracy* measures the performance of a classification model as the number of correct 
predictions divided by the total number of predictions. Its main advantage is its easy interpretability. 
Nevertheless, *accuracy* should only be used for balanced datasets. When dealing with imbalanced datasets,
i.e. when some classes are much more frequent than others, *accuracy* is not a reliable performance measure. 

\begin{equation}
f_1 = 2 * \frac{Precision * Recall}{Precision + Recall}
\end{equation}

The $f_1$ Score is the harmonic mean of *precision* and *recall*, i.e. it applys equal weight to both. 
The $f_1$ Score represents a meaningful evaluation for imbalanced datasets. As such, we recommend to
choose `accuracy_score` for balanced datasets and `f1_score` for imbalanced datasets.

Additionally, for imbalanced datasets, i.e. situations in which the `f1_score` is chosen, the user should differentiate
between binary and multi-class classification. For multi-class classification, the 
parameter *average* ought to be specified, as its default is only applicable if targets are
[binary](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score).
Four other parameter values are possible: *micro*, *macro*, *weighted* and *samples*. *Samples* is only 
meaningful for multilabel classification, which will not be in the scope of this assignment. Thus, we will 
only examine *micro*, *macro* and *weighted*. 

The *macro* $f_1$ *score* is computed as a simple arithmetic mean of the per-class $f_1$ *scores*. 
It does not take label imbalance into account.

The *weighted* $f_1$ *score* alters *macro* to account for label imbalance. The weight is applied by 
the number of true instances for each label.

The *micro* $f_1$ *score* is calculated counting the total true positives, false negatives and false positives.
Thus, the *micro* $f_1$ *score* is equal to total number of true positives over the total number of all observations.
Further explanations can be found
[here](https://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification.).

In conclusion, we recommend to chose `average = weighted` for the performance metric `f1_score` for the 
purpose of this assignment as this will account for the imbalance in the dataset. 

The chosen metric used for our baseline should be inputted in the following prompt. Be sure to insert it 
like `accuracy_score` if you want to use `accuracy_score` or respective for all other metrics.""")
        
        while True:
            self.metric = input(prompt = "Please insert your metric here: ")
            if self.metric not in ["accuracy_score","balanced_accuracy_score", "top_k_accuracy_score",
                               "average_precision_score" , "brier_score_loss", "f1_score",  "log_loss",
                               "precision_score", "recall_score", "jaccard_score", "roc_auc_score"]:
                self._printmd("""You have chosen a metric which is not available here. Please see this
                list for further info, please input the metric exactly like depicted here:
                `accuracy_score`,`balanced_accuracy_score`, `top_k_accuracy_score`, `average_precision_score`,
                `brier_score_loss`, `f1_score`,  `log_loss`, `precision_score`, `recall_score`, `jaccard_score`,
                `roc_auc_score`.""")
                continue
            if self.metric == "f1_score":
                self._printmd("""You have selected the `f1_score`. Since the `f1_score` can have different
                usages as described earlier, depending on its inputs please input some keyword-arguments into
                the next prompt.
                Please be advised, that this is done via `average="weighted", ...`, meaning with a = as separator.""")
                while True:
                    self.kwargs = input(prompt = "Please enter your keyword arguments here: ")
                    try:
                        keyword, sep, value = self.kwargs.partition('=')
                        self.kwargs = {keyword: value.strip('"')}
                        self.baseline = getattr(metrics, self.metric)(self.mod.predict(self.X_test), self.y_test,
                                                                      **self.kwargs)
                        break
                    except ValueError as e:
                        print(e)
                        continue
            else:
                self.baseline = getattr(metrics, self.metric)(self.mod.predict(self.X_test), self.y_test)
            if text:
                self._printmd(
    """Awesome, you have set your baseline! Now call the function `apply_random_projection` to check out,
    how good your model performs when we reduce its dimensions.""")
            break
        
    def apply_random_projection(self, model = None, text: bool = True, **kwargs):
        if text:
            self._printmd(r"""
Random Projection is a dimensionality reduction technique which is based on the **Johnson-Lindenstrauss lemma**.
This method projects or transforms the higher dimensional data to a lower dimensional subspace.
It approximately preserves the pairwise distances of the data points. 
It uses a random matrix to perform the projection and hence the name random projection. 
This matrix is also sometimes refered to as map.

If the original dimension of data is $d$ and the target or projected dimension is $k$, where $k<<d$ 
then the random matrix is of size $k,\ d$. The random projection is explained below.

\begin{equation}
X_{k,\ N}^{RP} = R_{k,\ d} X_{d,\ N}
\end{equation}

Where

$X_{k,\ N}^{RP}$ is the random projected N observations in $k$ dimensions,

$R_{k,\ d}$ is the random matrix used for the projection or transformation,

$X_{d,\ N}$ is the original $N$ observations in d-dimension.

There are a few techniques to create the random matrix. Gaussian and Sparse are just 2 among them.

**Gaussian** – The random matrix is created in such a way that each entry is independently drawn from
the standard normal distribution $N(0, \frac{1}{n_{components}})$. Where $n_{components}$ is the dimensionality
of the target projection space.

**Sparse** – When a sparse matrix is used for the random projection to reduce the computational complexity, then this 
is a sparse projection. This is an alternate approach to Gaussian random projection matrix which ensures that a similar distance
between the observations is preserved while reducing the dimensions.

If the sparse projection matrix has $c$ nonzero entries per column, then the complexity of the operation 
is of order $O(ckN)$ instead of $O(dkN)$.

Now we can perform the same actions for our second dataset.""")
        self.accuracies = []
        self.dims = np.int32(np.linspace(1, self.data.shape[1], int(self.data.shape[1]/1)))
        # Loop over the projection sizes, k
        for dim in self.dims:
            # Create random projection
            sp = SparseRandomProjection(n_components = dim, random_state = 5)
            X = sp.fit_transform(self.X_train)

            # Train classifier of your choice on the sparse random projection
            self.model = model(**kwargs)
            self.model.fit(X, self.y_train)

            # Evaluate model and update accuracies
            test = sp.transform(self.X_test)
            self.accuracies.append(getattr(metrics, self.metric)(self.model.predict(test), self.y_test, **self.kwargs))
            
    def plot(self, title: str = ""):
        # Create figure
        plt.figure(figsize=(8,8))
        plt.xlabel("# of dimensions k")
        plt.ylabel(f"{self.metric}")
        plt.xlim([1, self.data.shape[1]])
        plt.ylim([0, 1])
        plt.title(title)

        # Plot baseline and random projection accuracies
        plt.plot(self.dims, [self.baseline] * len(self.accuracies), color = "r")
        plt.plot(self.dims, self.accuracies)
        plt.show()
        
    def prepare_fit(self, url: str, data_type: str = None, names: list = None, text: bool = False,
                    test_size = 0.3, standardize = True, columns_to_drop: list = None,
                    model = None, **kwargs):
        self.get_data(url = url, data_type = data_type, names = names, text = text,**kwargs)
        self.split_data(test_size = test_size, standardize = standardize, columns_to_drop = columns_to_drop,
                        text= text)
        self.baseline(model = model, text = text, **kwargs)
        self.apply_random_projection(model = model, text = text, **kwargs)
        
    
    def _jans_birthday(self):
        self.cost = 0
        self._printmd(
            """
# Birthday Present from Group 10

Dear Jan,

Group 10 (that includes Skyler MacGowan, Sebastian Sydow, Debasmita Dutta and Yannik Suhre), wishes you all the best for your
birthday! We hope you have/had a beautiful day despite these challenging times! As a small birthday present, we have programmed
a little riddle for you. Here you go:

A bat and a ball together cost 1.10€. The bat costs one euro more than the ball. Now our question for you is, how much costs
the ball? Please input, what you think into the prompt!
            """)
        
        counter = 0
        while True:
            self._riddle_for_jan()
            if "," in str(self.cost):
                self._printmd("""Got'cha! Be aware that this has to be a floating **point** number with a
                                 **point** as decimal seperator! Try again, this time with a **point** as decimal point! ;)""")
                counter += 1
                continue
            else:
                if self.cost == 0.1:
                    self._printmd("""Sorry, that is wrong. If you do the math, you will end with a total price of 1,20€ for
                                     bat and ball. That ain't work! Think again and try again. ;)""")
                    counter += 1
                    continue
                elif self.cost != 0.05:
                    self._printmd(f"""Sorry, your answer with {self.cost} is wrong. One hint,
                    try to solve the equation $x + (x + 1) = 1.1$. Try again.""")
                    counter += 1
                    continue
                elif self.cost == 0.05:
                    fun = input(prompt = f"Are you really want to log {self.cost} in? (yes/no) ")
                    if fun == "no":
                        self._printmd("""Hm, what shall we do with you? You do not wanna log the answer in... So we'd say,
                                      start anew :P""")
                        continue
                    elif fun == "yes":
                        counter += 1
                        self._printmd(f"""Boooooooyaaaaah! You got it right! It just only took you {counter} tries!""")
                        break
        print("                           !     !     ! \n\
(          (    *         |V|   |V|   |V|        )   *   )       ( \n\
 )   *      )             | |   | |   | |        (       (   *    ) \n\
(          (           (*******************)    *       *    )    * \n\
(     (    (           (    *         *    )               )    ( \n\
 )   * )    )          (   \|/       \|/   )         *    (      ) \n\
(     (     *          (<<<<<<<<<*>>>>>>>>>)               )    ( \n\
 )     )        ((*******************************))       (  *   ) \n\
(     (   *     ((         HAPPY BIRTHDAY!!!!    ))      * )    ( \n\
 ) *   )        ((   *    *   *    *    *    *   ))   *   (      ) \n\
(     (         ((  \|/  \|/ \|/  \|/  \|/  \|/  ))        )    ( \n\
*)     )        ((^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^))       (      ) \n\
(     (   (**********************************************) )  * (")
            
    def _riddle_for_jan(self):
        self.cost = input(prompt = "How much does the ball cost? ")
        if "." in self.cost:
            self.cost = float(self.cost)
    
    def _printmd(self, string):
        return display(Markdown(string))