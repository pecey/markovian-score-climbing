import jax
import jax.numpy as np
from jax import random
import matplotlib.pyplot as plt
from jax.scipy.stats import norm
import pickle as pkl
import numpy as onp
from scipy.stats import norm as onorm
from jax.experimental.optimizers import adam
import argparse


class MSC:
    def __init__(self, seed, mu, log_sigma, n_latent):
        self.seed = seed
        self.key = random.PRNGKey(seed)
        self.mu = mu
        self.log_sigma = log_sigma
        self.step_size = 0.01
        self.n_latent = n_latent

    # Sample examples from proposal. Shape of output : (n_latent, n_samples)
    def sample_from_proposal(self, n_samples):
        self.key, subkey = random.split(self.key)
        noise = random.normal(subkey, shape=(self.n_latent, n_samples))
        return self.mu.reshape(-1, 1) + np.exp(self.log_sigma).reshape(-1, 1) * noise

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
        return np.sum(norm.logpdf(z, loc=mu.reshape(-1, 1), scale=np.exp(log_sigma).reshape(-1, 1)), axis=0)

    # Log of the likelihood: log P(y|x, z)
    # SF : Survival Function = 1-CDF
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    def log_likelihood(self, y, x, z):
        return np.sum(y * norm.logcdf(np.dot(x, z)) + (1 - y) * onorm.logsf(np.dot(x, z)), axis=0)

        # Conditional Importance Sampling

    def cis(self, z_old, n_samples, x, y):
        # Sample n examples and replace the first example using z_old
        z = self.sample_from_proposal(n_samples)
        # Replace the first sample of every latent variable with the conditional sample
        z = jax.ops.index_update(z, jax.ops.index[:, 0], z_old)

        # Compute importance weights : w = p(z) p(y|z,x)/q(z)
        # TODO: What should be the size of these arrays?
        # Size of log_w : (n_latent, 1)
        log_w = self.log_prior(z) + self.log_likelihood(y, x, z) - self.log_proposal(z, self.mu, self.log_sigma)
        max_log_w = np.max(log_w)
        shifted_w = np.exp(log_w - max_log_w)
        importance_weights = shifted_w / np.sum(shifted_w)

        # Sample next conditional sample
        j = self.sample_according_to_weights(importance_weights)
        return z, z[:, j], importance_weights

    def objective(self, importance_weights, z, mu, log_sigma):
        return -np.sum(importance_weights * self.log_proposal(z, mu, log_sigma))

    # # https://jax.readthedocs.io/en/latest/jax.experimental.optimizers.html
    # def step(self, step, opt_state, opt_update, importance_weights, z):
    #     value, gradient = jax.value_and_grad(self.objective, (2, 3))(importance_weights, z, self.mu, self.log_sigma)
    #     return opt_update(step, gradient, opt_state), value

    def step(self, step, importance_weights, z):
        value, gradient = jax.value_and_grad(self.objective, (2, 3))(importance_weights, z, self.mu, self.log_sigma)
        learning_rate = self.step_size / float(step + 1)
        self.mu = self.mu - learning_rate * gradient[0]
        self.log_sigma = self.log_sigma - learning_rate * gradient[1]
        return value

    def msc(self, train_x, train_y, n_samples = 10,  n_iterations = 10000, log_frequency = 500):
        conditional_sample = 0.1 * onp.random.normal(size=self.n_latent)
        # opt_init, opt_update, get_params = adam(self.s)
        # opt_state = opt_init((self.mu, self.log_sigma))
        for k in range(n_iterations):
            z, conditional_sample, importance_weights = self.cis(conditional_sample, n_samples, train_x, train_y)
            # Compute derivative wrt mu and log_sigma
            # opt_state, value = self.step(k, opt_state, opt_update, importance_weights, z)
            value = self.step(k, importance_weights, z)

            if k % log_frequency == 0:
                print(f"Iteration: {k}, Objective Value : {value}")

        return self.mu, self.log_sigma


def train_test_split(features_data, target_data, test_percentage=0.1):
    n_examples = features_data.shape[0]
    n_test = int(n_examples * test_percentage)

    all_indices = list(range(n_examples))
    test_indices = onp.random.choice(all_indices, size=n_test)
    train_indices = list(set(all_indices) - set(test_indices))

    train_x = features_data[train_indices]
    train_y = target_data[train_indices]

    test_x = features_data[test_indices]
    test_y = target_data[test_indices]

    return (train_x, train_y), (test_x, test_y)


# https://rpubs.com/cakapourani/variational-bayes-bpr
def evaluate(x, y, mu, log_sigma):
    variance = np.diag(np.exp(2 * log_sigma))
    predictive_prob = norm.cdf(np.dot(x, mu) / np.sqrt(1 + np.sum(np.dot(x, variance) * x, axis = 1)))
    prediction = (predictive_prob > 0.5).astype('float').reshape(-1, 1)
    test_error = 1 - np.sum(prediction == y) / len(y)
    return test_error

def main(args):
    # Read in the data
    features_data = onp.loadtxt(args.file_path, delimiter=',', usecols=range(0, 34))
    target_data = onp.loadtxt(args.file_path, delimiter=',', usecols=34, dtype='str')
    target_data = (target_data == 'g').astype('float').reshape(-1, 1)

    n_latent = features_data.shape[1]

    for i in range(args.n_experiments):
        # Train and test split
        (train_x, train_y), (test_x, test_y) = train_test_split(features_data, target_data)

        # Parameters of proposal distribution
        initial_mu, initial_log_sigma = onp.random.normal(size=(n_latent)), onp.random.normal(
            size=(n_latent))
        msc = MSC(seed=args.seed, mu=initial_mu, log_sigma=initial_log_sigma, n_latent=n_latent)
        mu, log_sigma = msc.msc(train_x, train_y, n_samples=args.n_samples, n_iterations = args.n_iterations)
        test_error = evaluate(test_x, test_y, mu, log_sigma)
        print(f"Test error: {test_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--file_path', type=str, help='Path of Ionos data file')
    parser.add_argument('--n_samples', type=int, help='Number of samples to sample from proposal', default=10)
    parser.add_argument('--n_iterations', type=int, help='Number of gradient steps to run', default=10000)
    parser.add_argument('--n_experiments', type=int, help='Number of times to run the experiment', default=10)
    parser.add_argument('--seed', type=int, help='Seed RNG', default=42)
    args = parser.parse_args()
    main(args)
