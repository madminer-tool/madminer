from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import logging
from scipy.stats import norm

# MadMiner output
logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.WARNING
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)

from madminer.ml import ParameterizedRatioEstimator

if not os.path.exists("tests/data"):
    os.makedirs("tests/data")


# Simulator settings
z_std = 2.0
x_std = 1.0


# Simulator
def simulate(theta, theta0=None, theta1=None, theta_score=None, npoints=None):
    # Draw latent variable
    z = np.random.normal(loc=theta, scale=z_std, size=npoints)

    # Draw observable
    x = np.random.normal(loc=z, scale=x_std, size=None)

    # Calculate joint likelihood ratio and joint score
    if theta0 is not None and theta1 is not None:
        r_xz = norm(loc=theta0, scale=z_std).pdf(z) / norm(loc=theta1, scale=z_std).pdf(z)
    else:
        r_xz = None

    if theta_score is not None:
        t_xz = (x - theta_score) / z_std ** 2
    else:
        t_xz = None

    return x, r_xz, t_xz


# True likeleihood ratio function
def calculate_likelihood_ratio(x, theta0, theta1=0.0):
    combined_std = (z_std ** 2 + x_std ** 2) ** 0.5
    r_x = norm(loc=theta0, scale=combined_std).pdf(x) / norm(loc=theta1, scale=combined_std).pdf(x)
    return r_x


def generate_data(sample_sizes):
    # Run simulator and generate etraining data
    n_param_points = max(sample_sizes) // 2  # number of parameter points to train

    theta0 = np.random.uniform(low=-4.0, high=4.0, size=n_param_points)  # numerator, uniform prior
    theta1 = np.zeros(shape=n_param_points)  # denominator: fixed at 0

    # Sample from theta0
    x_from_theta0, r_xz_from_theta0, t_xz_from_theta0 = simulate(theta0, theta0, theta1, theta0)

    # Sample from theta1
    x_from_theta1, r_xz_from_theta1, t_xz_from_theta1 = simulate(theta1, theta0, theta1, theta0)

    # Combine results and reshape
    x_train = np.hstack((x_from_theta0, x_from_theta1)).reshape(-1, 1)
    r_xz_train = np.hstack((r_xz_from_theta0, r_xz_from_theta1)).reshape(-1, 1)
    t_xz_train = np.hstack((t_xz_from_theta0, t_xz_from_theta1)).reshape(-1, 1)
    y_train = np.hstack((np.zeros_like(x_from_theta0), np.ones_like(np.ones_like(x_from_theta1)))).reshape(-1, 1)
    theta0_train = np.hstack((theta0, theta0)).reshape(-1, 1)

    # Save everything to files.
    np.save("tests/data/theta0_train.npy", theta0_train)
    np.save("tests/data/x_train.npy", x_train)
    np.save("tests/data/y_train.npy", y_train)
    np.save("tests/data/r_xz_train.npy", r_xz_train)
    np.save("tests/data/t_xz_train.npy", t_xz_train)


def run_test(method, alpha, sample_size):
    # Train model
    estimator = ParameterizedRatioEstimator(n_hidden=(50, 50))
    estimator.train(
        method=method,
        x="tests/data/x_train.npy",
        y="tests/data/y_train.npy",
        theta="tests/data/theta0_train.npy",
        r_xz="tests/data/r_xz_train.npy",
        t_xz="tests/data/t_xz_train.npy",
        alpha=alpha,
        limit_samplesize=sample_size,
        verbose="few",
    )

    # Generate evaluation data
    n_x_test = 50
    n_thetas_grid = 20

    theta_test = 1.0 * np.ones(shape=n_x_test).reshape(-1, 1)
    x_test, _, _ = simulate(theta_test)
    np.save("tests/data/x_test.npy", x_test)

    # We want to evaluate the expected likelihood ratio on a range of parameter points
    theta_grid = np.linspace(-4.0, 4.0, n_thetas_grid).reshape(-1, 1)
    np.save("tests/data/theta_grid.npy", theta_grid)

    # Ground truth
    log_r_test_true = []
    for theta in theta_grid:
        log_r_test_true.append(np.log(calculate_likelihood_ratio(x_test, theta)))
    log_r_test_true = np.array(log_r_test_true).reshape(n_thetas_grid, n_x_test)

    # Evaluation
    log_r_tests_alices, _ = estimator.evaluate(
        theta="tests/data/theta_grid.npy", x="tests/data/x_test.npy", evaluate_score=False
    )

    # Calculate error
    rmse = np.mean((log_r_test_true - log_r_tests_alices) ** 2)

    return rmse


def test_ratio_estimation():
    methods = ["carl", "rolr", "alice", "cascal", "rascal", "alices"]
    alphas = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    sample_sizes = [100, 10000]
    rmses = []

    print("Generating data for ratio estimation")
    generate_data(sample_sizes)

    for method, alpha in zip(methods, alphas):
        this_rmses = []
        for sample_size in sample_sizes:
            print("Training method {} on {} samples".format(method, sample_size))
            this_rmses.append(run_test(method, alpha, sample_size))
            print("  -> MSE =", this_rmses[-1])
        rmses.append(this_rmses)
    rmses = np.asarray(rmses)

    print("")
    print("Results: Mean squared error of log r")
    print("")
    print(" Method  |  100 samples  |  10k samples ")
    print("------------------------------------------")
    for method, this_rmses in zip(methods, rmses):
        print(" {:>6s}  |    {:11.3f}  |  {:11.3f} ".format(method, this_rmses[0], this_rmses[1]))

    print("")

    assert np.max(rmses[:, -1]) < 100.0


if __name__ == "__main__":
    test_ratio_estimation()
