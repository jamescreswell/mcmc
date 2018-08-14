# File containing classes and functions for MCMC sampling

import numpy as np
import matplotlib.pyplot as plt
import random
import math


# Example usage:
# data = [data_x, data_y]

# def log_prior(theta):
#     nu, a, b = theta
#     if nu < 0:
#         return -inf
#     else:
#         return 1

# def log_likelihood(data, theta):
#     x, y = data
#     nu, a, b = theta
#     return f(x, y, nu, a, b)

# sampler = Metropolis(data, log_prior, log_likelihood)

class Metropolis:
    def __init__(self, data, log_prior, log_likelihood):
        self.data = data
        self.log_prior = log_prior 
        self.log_likelihood = log_likelihood
        
        self.chain = np.array([])
       
    def iterate(self, start, n_iterations, proposal_mean, proposal_std):
        """Iterates the chain n_iterations times"""
        i = 0
        self.chain = [np.array(start)]
        while i < n_iterations:
            theta_proposed = self.chain[-1] + np.random.normal(proposal_mean, proposal_std, size=np.size(self.chain[-1]))
            logr = self.log_prior(theta_proposed) + self.log_likelihood(self.data, theta_proposed) - self.log_prior(self.chain[-1]) - self.log_likelihood(self.data, self.chain[-1])
            if logr > 0: 
                self.chain.append(theta_proposed)
            else:
                if math.log(random.random()) < logr:
                    self.chain.append(theta_proposed)
                else:
                    self.chain.append(self.chain[-1])
            i = i + 1
        self.chain = np.array(self.chain)
        
    def plot_histogram(self, axis, pburn, nbins=10, true=None):
        nburn = int(pburn*len(self.chain))
        
        plt.figure()
        plt.hist(self.chain[nburn:][:,axis], bins=nbins)
        if true:
            plt.axvline(true, color='red')