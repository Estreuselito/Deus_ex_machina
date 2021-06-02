#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from IPython.display import clear_output, display, Markdown
import time


# In[8]:


def em_algorithm(prior_a, prior_b, iterations, trials = 5):
    # create some binomial data
    n, p = 10, .5  # number of trials, probability of each trial
    # s is an array of 1000 trials, where each number represents the number of heads, which was thrown with
    # a probability of 0.5
#     s = np.random.binomial(n, p, trials)
    s = np.array([5, 9, 8, 4, 7])
    for i in range(0,iterations):
        probability_coin_a, probability_coin_b = [], []
        display(Markdown(f"We are currently at the {i} iteration. Our current probabilites for coin A & B are {prior_a} and {prior_b}."))
        for heads in s:
            coin_a = prior_a**heads * (1-prior_a)**(n-heads) / (prior_a**heads * (1-prior_a)**(n-heads) + 0.5**heads * 0.5**(n-heads))
            display(Markdown(f"We have {heads} appearences of heads and {n-heads} appearances of tails. Now we calculate             the probability that this data belongs to coin A using             $$\\frac{{{{{prior_a}^{heads}}}*{{{1-prior_a}^{n-heads}}}}}{{{{{{{prior_a}^{heads}}}*{{{1-prior_a}^{n-heads}}}}}+{{{{{0.5}^{heads}}}*{{{0.5}^{n-heads}}}}}}} = {round(coin_a,2)}$$"))
            display(Markdown(f"Our probability that this dataset belongs to coin A is {round(coin_a,2)},             that it belongs to coin b {round(1-coin_a,2)}."))
            probability_coin_a.append(coin_a)
            probability_coin_b.append(1-coin_a)
        coin_a_sum_heads = np.sum(probability_coin_a * s)
        coin_a_sum_tails = np.sum(probability_coin_a * (n-s))
        coin_b_sum_heads = np.sum(probability_coin_b * s)
        coin_b_sum_tails = np.sum(probability_coin_b * (n-s))
        prior_a = round(coin_a_sum_heads / (coin_a_sum_heads + coin_a_sum_tails),2)
        prior_b = round(coin_b_sum_heads / (coin_b_sum_heads + coin_b_sum_tails),2)
        display(Markdown(f"Now we updated our priors using the EM algorithm to coin A         {round(prior_a,2)} and coin B {round(prior_b,2)}."))
        time.sleep(10)
        clear_output()
    return round(prior_a,2), round(prior_b,2)


# In[9]:


em_algorithm(0.6, 0.4, 10)


# In[3]:


from sklearn.mixture import GaussianMixture
import numpy as np


# In[7]:


s = np.array([5, 9, 8, 4, 7]).reshape(-1, 1)


# In[17]:


# fit model
model = GaussianMixture(n_components=2, weights_init=(0.6,0.4), max_iter = 100)
model.fit(s)
# predict latent values
yhat = model.predict_proba(s)


# In[18]:


yhat.mean(axis=0)


# In[ ]:




