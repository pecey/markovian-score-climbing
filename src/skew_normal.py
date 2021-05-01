import jax
import jax.numpy as np
from jax import random
from jax.scipy.stats import norm
import argparse
import numpy as onp

# Parameters of true distribution
location, scale, shape = 0.5, 2., 5.
# https://en.wikipedia.org/wiki/Skew_normal_distribution
# Log of skew normal distribution
def log_p(x):
    scaled_x = (x - location) / scale
    return np.log(2) + norm.logcdf(scaled_x * shape) + norm.logpdf(scaled_x) - np.log(scale)

# Expectation of mean and variance
# https://en.wikipedia.org/wiki/Skew_normal_distribution
delta = shape / np.sqrt(1 + shape ** 2)
expected_mean = location + scale * delta * np.sqrt(2 / np.pi)
expected_variance = scale ** 2 * (1 - (2 * delta ** 2 / np.pi))


class MSC:
    def __init__(self, seed, log_p_fn, mu, log_sigma):
        self.seed = seed
        self.key = random.PRNGKey(seed)
        self.log_p = log_p_fn
        self.mu = mu
        self.log_sigma = log_sigma
        self.step_size = 0.5

    # Log of proposal distribution
    def log_q(self, x, mu, log_sigma):
        sigma = np.exp(log_sigma)
        scaled_x = (x - mu) / sigma
        return -0.5 * (scaled_x ** 2) - log_sigma - 0.5 * np.log(2 * np.pi)

    def sample_from_proposal(self, n_samples):
        self.key, subkey = random.split(self.key)
        noise = random.normal(subkey, shape=(n_samples,))
        return self.mu + np.exp(self.log_sigma) * noise

    # Randomly sample 1..N according to weights
    def sample_according_to_weights(self, weights):
        self.key, subkey = random.split(self.key)
        x = random.uniform(subkey)
        bins = np.cumsum(weights)
        return np.digitize(x, bins)

    # Conditional Importance Sampling
    def cis(self, z_old, n_samples):
        # Sample n examples and replace the first example using z_old
        z = self.sample_from_proposal(n_samples)
        if z_old is not None:
            # JAX equivalent of z[0]=z_old
            z = jax.ops.index_update(z, 0, z_old)

        # Compute importance weights.
        log_w = self.log_p(z) - self.log_q(z, self.mu, self.log_sigma)
        max_log_w = np.max(log_w)
        shifted_w = np.exp(log_w - max_log_w)
        importance_weights = shifted_w / np.sum(shifted_w)

        if z_old is not None:
            # Sample next conditional sample
            j = self.sample_according_to_weights(importance_weights)
            return z, z[j], importance_weights
        else:
            return z, None, importance_weights

    def objective(self, importance_weights, z, mu, log_sigma):
        return -np.sum(importance_weights * self.log_q(z, mu, log_sigma))

    def approximate(self, cis = True, random_init = True):
        log_frequency = 5000
        n_iterations = 1000000
        n_samples = 2
        conditional_sample = 0 if cis else None
        for k in range(n_iterations):
            z, conditional_sample, importance_weights = self.cis(conditional_sample, n_samples)
            # Compute derivative wrt mu and log_sigma
            gradient = jax.grad(self.objective, (2, 3))(importance_weights, z, self.mu, self.log_sigma)
            # Gradient Step
            self.mu = self.mu - self.step_size * gradient[0] / float(k + 1)
            self.log_sigma = self.log_sigma - self.step_size * gradient[1] / float(k + 1)
            if k % log_frequency == 0:
                print(f"Iteration: {k}, Mean: {self.mu}, Log Sigma: {self.log_sigma}")

def main(args):
    cis = args.cis.lower() == "true"
    # Parameters of proposal distribution
    # mu, log_sigma = expected_mean, np.log(np.sqrt(expected_variance))
    # print(f"Initial parameters : Mu : {mu}, Log sigma : {log_sigma}")
    onp.random.seed(args.seed)
    mu, log_sigma = onp.random.random(), onp.random.random()
    alg = MSC(args.seed, log_p, mu, log_sigma)
    alg.approximate(cis = cis)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--cis', type=str, help='Whether to run conditional IS or IS', default="true")
    parser.add_argument('--seed', type=int, help='Seed RNG', default=42)
    args = parser.parse_args()
    main(args)

