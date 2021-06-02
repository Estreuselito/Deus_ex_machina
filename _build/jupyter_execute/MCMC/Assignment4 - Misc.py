#!/usr/bin/env python
# coding: utf-8

# With the introduction of noise to the data, i.e. through $\sigma_{Recovered}$, $\sigma_{Infected}$ and $\sigma_{Dead}$, individual data points may exceed the value of 1.0. Furthermore, for a certain point in time the sum of R(t), I(t) and D(t) may exceed the amount of 1.0. This contradicts reality as the size of the population can not exceed 100%, i.e. 1. Thus, the compartments *recovered*, *infected* and *dead* should maximally yield a sum of 1, which in turn corresponds to the amount of *suceptibles* being 0. In order to address this issue, the modelling of noise would need to be adjusted, e.g. by introducing a constraint. As this did not significantly effect the second part of our assignment, we did not chose to alter the modelling of noise. Nevertheless, it is important to keep this shortcoming in mind if applied to other tasks.

# # Key Properties of Markov Chains
# **Note**: Normalization, Ergodicity, and Homogeneity are *general properties*, whereas Reversibility is a *special property*. A general property is one that is typically assumed to be true unless specified/proved otherwise, while the opposite is the case for special properties.
# 
# 
# ### Normalization
# For each state, the transition probabilities sum to 1.
# 
# ### Ergodicity
# Ergodicity expresses the idea that a point of a moving system, either a dynamical system or a stochastic process, will eventually (i.e. in a finite number of steps and with a positive probability) visit all parts of the space that the system moves in, in a uniform and random sense. [Class Notes, 1] 
# 
# ### Homogeneity
# A Markov chain is called homogeneous if and only if the transition probabilities are independent of the time. [2]
# 
# 
# ### Reversible 
# Reversibility is obtained if the chain is symmetric, meaning that running it backwards is equivalent to running it forwards.
# 
# ## Sources
# 
# [1] University of Wisconson: https://people.math.wisc.edu/~valko/courses/331/MC2.pdf \
# [2] Texas A&M University: https://people.engr.tamu.edu/andreas-klappenecker/csce658-s18/markov_chains.pdf
