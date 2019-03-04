from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import torch
from torch import tensor

from madminer.utils.ml.models.ratio import DenseSingleParameterizedRatioModel, DenseDoublyParameterizedRatioModel

logger = logging.getLogger(__name__)


def evaluate_flow_model(model, thetas=None, xs=None, evaluate_score=False, run_on_gpu=True, double_precision=False):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Balance theta0 and theta1
    n_thetas = len(thetas)

    # Prepare data
    n_xs = len(xs)
    thetas = torch.stack([tensor(thetas[i % n_thetas], requires_grad=True) for i in range(n_xs)])
    xs = torch.stack([tensor(i) for i in xs])

    model = model.to(device, dtype)
    thetas = thetas.to(device, dtype)
    xs = xs.to(device, dtype)

    # Evaluate estimator with score:
    if evaluate_score:
        model.eval()

        _, log_p_hat, t_hat = model.log_likelihood_and_score(thetas, xs)

        # Copy back tensors to CPU
        if run_on_gpu:
            log_p_hat = log_p_hat.cpu()
            t_hat = t_hat.cpu()

        log_p_hat = log_p_hat.detach().numpy().flatten()
        t_hat = t_hat.detach().numpy().flatten()

    # Evaluate estimator without score:
    else:
        with torch.no_grad():
            model.eval()

            _, log_p_hat = model.log_likelihood(thetas, xs)

            # Copy back tensors to CPU
            if run_on_gpu:
                log_p_hat = log_p_hat.cpu()

            log_p_hat = log_p_hat.detach().numpy().flatten()
            t_hat = None

    return log_p_hat, t_hat


def evaluate_ratio_model(
    model,
    method_type=None,
    theta0s=None,
    theta1s=None,
    xs=None,
    evaluate_score=False,
    run_on_gpu=True,
    double_precision=False,
    return_grad_x=False,
):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Figure out method type
    if method_type is None:
        if isinstance(model, DenseSingleParameterizedRatioModel):
            method_type = "parameterized"
        elif isinstance(model, DenseDoublyParameterizedRatioModel):
            method_type = "doubly_parameterized"
        else:
            raise RuntimeError("Cannot infer method type automatically")

    # Balance theta0 and theta1
    if theta1s is None:
        n_thetas = len(theta0s)
    else:
        n_thetas = max(len(theta0s), len(theta1s))
        if len(theta0s) > len(theta1s):
            theta1s = np.array([theta1s[i % len(theta1s)] for i in range(len(theta0s))])
        elif len(theta0s) < len(theta1s):
            theta0s = np.array([theta0s[i % len(theta0s)] for i in range(len(theta1s))])

    # Prepare data
    n_xs = len(xs)
    theta0s = torch.stack([tensor(theta0s[i % n_thetas], requires_grad=evaluate_score) for i in range(n_xs)])
    if theta1s is not None:
        theta1s = torch.stack([tensor(theta1s[i % n_thetas], requires_grad=evaluate_score) for i in range(n_xs)])
    xs = torch.stack([tensor(i) for i in xs])

    model = model.to(device, dtype)
    theta0s = theta0s.to(device, dtype)
    if theta1s is not None:
        theta1s = theta1s.to(device, dtype)
    xs = xs.to(device, dtype)

    # Evaluate ratio estimator with score or x gradients:
    if evaluate_score or return_grad_x:
        model.eval()

        if method_type == "parameterized_ratio":
            if return_grad_x:
                s_hat, log_r_hat, t_hat0, x_gradients = model(
                    theta0s, xs, return_grad_x=True, track_score=evaluate_score, create_gradient_graph=False
                )
            else:
                s_hat, log_r_hat, t_hat0 = model(theta0s, xs, track_score=evaluate_score, create_gradient_graph=False)
                x_gradients = None
            t_hat1 = None
        elif method_type == "double_parameterized_ratio":
            if return_grad_x:
                s_hat, log_r_hat, t_hat0, t_hat1, x_gradients = model(
                    theta0s, theta1s, xs, return_grad_x=True, track_score=evaluate_score, create_gradient_graph=False
                )
            else:
                s_hat, log_r_hat, t_hat0, t_hat1 = model(
                    theta0s, theta1s, xs, track_score=evaluate_score, create_gradient_graph=False
                )
                x_gradients = None
        else:
            raise ValueError("Unknown method type %s", method_type)

        # Copy back tensors to CPU
        if run_on_gpu:
            s_hat = s_hat.cpu()
            log_r_hat = log_r_hat.cpu()
            if t_hat0 is not None:
                t_hat0 = t_hat0.cpu()
            if t_hat1 is not None:
                t_hat1 = t_hat1.cpu()

        # Get data and return
        s_hat = s_hat.detach().numpy().flatten()
        log_r_hat = log_r_hat.detach().numpy().flatten()
        if t_hat0 is not None:
            t_hat0 = t_hat0.detach().numpy()
        if t_hat1 is not None:
            t_hat1 = t_hat1.detach().numpy()

    # Evaluate ratio estimator without score:
    else:
        with torch.no_grad():
            model.eval()

            if method_type == "parameterized_ratio":
                s_hat, log_r_hat, _ = model(theta0s, xs, track_score=False, create_gradient_graph=False)
            elif method_type == "double_parameterized_ratio":
                s_hat, log_r_hat, _, _ = model(theta0s, theta1s, xs, track_score=False, create_gradient_graph=False)
            else:
                raise ValueError("Unknown method type %s", method_type)

            # Copy back tensors to CPU
            if run_on_gpu:
                s_hat = s_hat.cpu()
                log_r_hat = log_r_hat.cpu()

            # Get data and return
            s_hat = s_hat.detach().numpy().flatten()
            log_r_hat = log_r_hat.detach().numpy().flatten()
            t_hat0, t_hat1 = None, None

    if return_grad_x:
        return s_hat, log_r_hat, t_hat0, t_hat1, x_gradients
    return s_hat, log_r_hat, t_hat0, t_hat1


def evaluate_local_score_model(model, xs=None, run_on_gpu=True, double_precision=False, return_grad_x=False):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Prepare data
    xs = torch.stack([tensor(i) for i in xs])

    model = model.to(device, dtype)
    xs = xs.to(device, dtype)

    # Evaluate networks
    if return_grad_x:
        model.eval()
        t_hat, x_gradients = model(xs, return_grad_x=True)
    else:
        with torch.no_grad():
            model.eval()
            t_hat = model(xs)
        x_gradients = None

    # Copy back tensors to CPU
    if run_on_gpu:
        t_hat = t_hat.cpu()
        if x_gradients is not None:
            x_gradients = x_gradients.cpu()

    # Get data and return
    t_hat = t_hat.detach().numpy()

    if return_grad_x:
        x_gradients = x_gradients.detach().numpy()
        return t_hat, x_gradients

    return t_hat
