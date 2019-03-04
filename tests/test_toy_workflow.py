from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging
import numpy as np
from scipy.stats import norm

from madminer.ml import MLForge

if not os.path.exists("tests/data"):
    os.makedirs("tests/data")


# Simulator settings
def z_mean(theta):
    return theta + 10.0


z_std = 2.0
x_std = 1.0


# Simulator
def simulate(theta, theta0=None, theta1=None, theta_score=None, npoints=None):
    # Draw latent variable
    z = np.random.normal(loc=z_mean(theta), scale=z_std, size=npoints)

    # Draw observable
    x = np.random.normal(loc=z, scale=x_std, size=None)

    # Calculate joint likelihood ratio and joint score
    if theta0 is not None and theta1 is not None:
        r_xz = norm(loc=z_mean(theta0), scale=z_std).pdf(z) / norm(loc=z_mean(theta1), scale=z_std).pdf(z)
    else:
        r_xz = None

    if theta_score is not None:
        t_xz = (x - z_mean(theta_score)) / z_std ** 2
    else:
        t_xz = None

    return x, r_xz, t_xz


# True likeleihood ratio function
def calculate_likelihood_ratio(x, theta0, theta1=0.0):
    combined_std = (z_std ** 2 + x_std ** 2) ** 0.5
    r_x = norm(loc=z_mean(theta0), scale=combined_std).pdf(x) / norm(loc=z_mean(theta1), scale=combined_std).pdf(x)
    return r_x


def run_test():
    # Run simulator and generate etraining data
    n_param_points = 5000  # number of parameter points to train

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

    # Train model
    forge = MLForge()

    forge.train(
        method="alices",
        x_filename="tests/data/x_train.npy",
        y_filename="tests/data/y_train.npy",
        theta0_filename="tests/data/theta0_train.npy",
        r_xz_filename="tests/data/r_xz_train.npy",
        t_xz0_filename="tests/data/t_xz_train.npy",
        alpha=0.1,
        n_epochs=10,
        n_hidden=(20, 20),
        validation_split=None,
        batch_size=256,
    )

    # Generate evaluation data
    n_param_points_test = 100
    n_thetas_grid = 100

    theta_test = 1.0 * np.ones(shape=n_param_points_test).reshape(-1, 1)
    x_test, _, _ = simulate(theta_test)
    np.save("tests/data/x_test.npy", x_test)

    # We want to evaluate the expected likelihood ratio on a range of parameter points
    theta_grid = np.linspace(-4.0, 4.0, n_thetas_grid).reshape(-1, 1)
    np.save("tests/data/theta_grid.npy", theta_grid)

    # Ground truth
    log_r_test_true = []
    for theta in theta_grid:
        log_r_test_true.append(np.log(calculate_likelihood_ratio(x_test, theta)))
    log_r_test_true = np.array(log_r_test_true)

    # Evaluation
    log_r_tests_alices, _, _ = forge.evaluate(
        theta0_filename="tests/data/theta_grid.npy", x="tests/data/x_test.npy", evaluate_score=False
    )

    # Calculate error
    rmse = np.mean((log_r_test_true - log_r_tests_alices) ** 2) ** 0.5

    return rmse


def test_toy_workflow():
    rmse = run_test()

    print("Root mean squared error of true log r vs ALICES log r: {}".format(rmse))

    assert rmse < 1000.0


if __name__ == "__main__":
    test_toy_workflow()
