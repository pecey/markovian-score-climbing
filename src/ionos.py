import jax
import jax.numpy as np
from jax import random, jit
import matplotlib.pyplot as plt
from jax.scipy.stats import norm
import pickle as pkl
import numpy as onp
from scipy.stats import norm as onorm
from jax.experimental.optimizers import adam
import argparse


class MSC:
    def __init__(self, seed, n_latent):
        self.seed = seed
        self.key = random.PRNGKey(seed)
        self.n_latent = n_latent
        self.step_size = 0.01
        self.opt_init, self.opt_update, self.get_params = adam(self.step_size)

    # Wrapper function
    def sample_from_normal(self, shape):
        if type(shape) == int:
            shape = (shape,)
        self.key, subkey = random.split(self.key)
        return random.normal(subkey, shape=shape)

    def init_params(self, random_init):
        if random_init:
            return self.sample_from_normal(shape=self.n_latent), self.sample_from_normal(shape=self.n_latent)
        return 0.1 * self.sample_from_normal(shape=self.n_latent), 0.5 + 0.1 * self.sample_from_normal(
            shape=self.n_latent)

    # Sample examples from proposal. Shape of output : (n_latent, n_samples)
    def sample_from_proposal(self, mu, log_sigma, n_samples):
        noise = self.sample_from_normal(shape=(self.n_latent, n_samples))
        return mu.reshape(-1, 1) + np.exp(log_sigma).reshape(-1, 1) * noise

    # Randomly sample 1..N according to weights
    def sample_according_to_weights(self, weights):
        self.key, subkey = random.split(self.key)
        x = random.uniform(subkey)
        bins = np.cumsum(weights)
        return np.digitize(x, bins)

    # Log of the prior: log P(z) where P(z) ~ N(0,1)
    def log_prior(self, z):
        return np.sum(norm.logpdf(z), axis=0)

    # Log of proposal distribution
    def log_proposal(self, z, mu, log_sigma):
        sigma = np.exp(log_sigma)
        return np.sum(norm.logpdf(z, loc=mu.reshape(-1, 1), scale=sigma.reshape(-1, 1)), axis=0)

    # Log of the likelihood: log P(y|x, z)
    # SF : Survival Function = 1-CDF
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    def log_likelihood(self, y, x, z):
        return np.sum(y * norm.logcdf(np.dot(x, z)) + (1 - y) * onorm.logsf(np.dot(x, z)), axis=0)

    # Conditional Importance Sampling
    def cis(self, mu, log_sigma, z_old, n_samples, x, y):
        # Sample n examples and replace the first example using z_old
        z = self.sample_from_proposal(mu, log_sigma, n_samples)
        # Replace the first sample of every latent variable with the conditional sample
        if z_old is not None:
            z = jax.ops.index_update(z, jax.ops.index[:, 0], z_old)

        # Compute importance weights : w = p(z) p(y|z,x)/q(z)
        # Size of log_w : (n_latent, 1)
        log_w = self.log_prior(z) + self.log_likelihood(y, x, z) - self.log_proposal(z, mu, log_sigma)
        max_log_w = np.max(log_w)
        shifted_w = np.exp(log_w - max_log_w)
        importance_weights = shifted_w / np.sum(shifted_w)

        if z_old is not None:
            # Sample next conditional sample
            j = self.sample_according_to_weights(importance_weights)
            return z, z[:, j], importance_weights
        else:
            return z, None, importance_weights

    def objective(self, importance_weights, z, mu, log_sigma):
        return -np.sum(importance_weights * self.log_proposal(z, mu, log_sigma))

    # # https://jax.readthedocs.io/en/latest/jax.experimental.optimizers.html
    def step(self, step, mu, log_sigma, importance_weights, z, opt_state):
        # Differentiate wrt 3rd and 4th parameter
        gradient = self.derivative_of_objective()(importance_weights, z, mu, log_sigma)
        opt_state = self.opt_update(step, gradient, opt_state)
        return opt_state, self.get_params(opt_state)

    def derivative_of_objective(self):
        return jax.jit(jax.grad(self.objective, (2, 3)))

    def init_conditional_sample(self, cis=True, random_init=False):
        if not cis:
            return None
        if random_init:
            return self.sample_from_normal(shape=self.n_latent)
        return 0.1 * self.sample_from_normal(shape=self.n_latent)

    def approximate(self, train_x, train_y, n_samples=10, n_iterations=1000, log_frequency=500, cis=False,
                    random_init=True):
        conditional_sample = self.init_conditional_sample(cis, random_init)
        mu, log_sigma = self.init_params(random_init)
        opt_state = self.opt_init((mu, log_sigma))
        mu_ = []
        log_sigma_ = []
        for k in range(n_iterations):
            z, conditional_sample, importance_weights = self.cis(mu, log_sigma, conditional_sample, n_samples, train_x,
                                                                 train_y)
            # Compute derivative wrt mu and log_sigma
            # opt_state, value = self.step(k, opt_state, opt_update, importance_weights, z)
            opt_state, (mu, log_sigma) = self.step(k, mu, log_sigma, importance_weights, z, opt_state)
            mu_.append(mu)
            log_sigma_.append(log_sigma)
            if k % log_frequency == 0:
                value = self.objective(importance_weights, z, mu, log_sigma)
                print(f"Iteration: {k}, Objective Value : {value}")

        return mu, log_sigma, mu_, log_sigma_


def train_test_split(features_data, target_data, test_percentage=0.1):
    n_examples = features_data.shape[0]
    n_test = int(n_examples * test_percentage)

    permuted_indices = onp.random.permutation(n_examples)
    test_indices = permuted_indices[:n_test]
    train_indices = permuted_indices[n_test:]

    train_x = features_data[train_indices]
    train_y = target_data[train_indices]

    test_x = features_data[test_indices]
    test_y = target_data[test_indices]

    return (train_x, train_y), (test_x, test_y)


# https://rpubs.com/cakapourani/variational-bayes-bpr
def evaluate(x, y, mu, variance):
    predictive_prob = norm.cdf(np.dot(x, mu) / np.sqrt(1 + np.sum(np.dot(x, variance) * x, axis=1)))
    prediction = (predictive_prob > 0.5).astype('float').reshape(-1, 1)
    test_error = 1 - np.sum(prediction == y) / len(y)
    return test_error

def main(args):

    # Read in the data
    features_data = onp.loadtxt(args.file_path, delimiter=',', usecols=range(0, 34))
    features_data = onp.insert(features_data, 0, 1, axis=1)
    target_data = onp.loadtxt(args.file_path, delimiter=',', usecols=34, dtype='str')
    target_data = (target_data == 'g').astype('float').reshape(-1, 1)

    n_latent = features_data.shape[1]

    conditional_importance_sampling = args.cis.lower() == "true"
    random_init = args.random_init.lower() == "true"

    print(f"Arguments: {args}, CIS: {conditional_importance_sampling}, Random Init: {random_init}")

    for i in range(args.n_experiments):
        # Train and test split
        (train_x, train_y), (test_x, test_y) = train_test_split(features_data, target_data)
        msc = MSC(seed=args.seed, n_latent=n_latent)
        mu, log_sigma, mu_history, log_sigma_history = msc.approximate(train_x, train_y, n_samples=args.n_samples, n_iterations = args.n_iterations, cis=conditional_importance_sampling, random_init=random_init)
        mu_opt = np.mean(np.array(mu_history[-150:]), axis = 0)
        var_opt = np.diag(np.mean(np.exp(2 * np.array(log_sigma_history[-150:])), axis=0))
        test_error = evaluate(test_x, test_y, mu_opt, var_opt)
        print(f"Test error: {test_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--file_path', type=str, help='Path of data file')
    parser.add_argument('--n_samples', type=int, help='Number of samples to sample from proposal', default=10)
    parser.add_argument('--n_iterations', type=int, help='Number of gradient steps to run', default=10000)
    parser.add_argument('--n_experiments', type=int, help='Number of times to run the experiment', default=10)
    parser.add_argument('--seed', type=int, help='Seed RNG', default=42)
    parser.add_argument('--cis', type=str, help='Whether to run conditional IS or IS', default="true")
    parser.add_argument('--random_init', type=str, help='Whether to run with random initialization or initialization in paper', default="true")
    args = parser.parse_args()
    main(args)
