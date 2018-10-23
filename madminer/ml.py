from __future__ import absolute_import, division, print_function

import six
import logging
import os
import json
import numpy as np

import torch

from madminer.utils.ml import ratio_losses, flow_losses
from madminer.utils.ml.models.maf import ConditionalMaskedAutoregressiveFlow
from madminer.utils.ml.models.maf_mog import ConditionalMixtureMaskedAutoregressiveFlow
from madminer.utils.ml.models.ratio import ParameterizedRatioEstimator, DoublyParameterizedRatioEstimator
from madminer.utils.ml.models.score import LocalScoreEstimator
from madminer.utils.ml.flow_trainer import train_flow_model, evaluate_flow_model
from madminer.utils.ml.ratio_trainer import train_ratio_model, evaluate_ratio_model
from madminer.utils.ml.score_trainer import train_local_score_model, evaluate_local_score_model
from madminer.utils.various import create_missing_folders, load_and_check, general_init


class MLForge:
    """
    Estimators for the likelihood ratio and score based on machine learning.

    Each instance of this class represents one neural estimator. The most important functions are:

    * `MLForge.train()` to train an estimator.

    * `MLForge.evaluate()` to evaluate the estimator.

    * `MLForge.save()` to save the trained model to files.

    * `MLForge.load()` to load the trained model from files.

    Please see the tutorial for a detailed walk-through.

    Parameters
    ----------
    debug : bool, optional
        If True, additional detailed debugging output is printed. Default value: False.

    """

    def __init__(self, debug=False):
        general_init(debug=debug)

        self.method_type = None
        self.model = None
        self.method = None
        self.nde_type = None
        self.n_observables = None
        self.n_parameters = None
        self.n_hidden = None
        self.activation = None
        self.maf_n_mades = None
        self.maf_batch_norm = None
        self.maf_batch_norm_alpha = None
        self.features = None

    def train(self,
              method,
              x_filename,
              y_filename=None,
              theta0_filename=None,
              theta1_filename=None,
              r_xz_filename=None,
              t_xz0_filename=None,
              t_xz1_filename=None,
              features=None,
              nde_type='maf',
              n_hidden=(100, 100, 100),
              activation='tanh',
              maf_n_mades=3,
              maf_batch_norm=True,
              maf_batch_norm_alpha=0.1,
              maf_mog_n_components=10,
              alpha=1.,
              n_epochs=20,
              batch_size=128,
              initial_lr=0.002,
              final_lr=0.0001,
              validation_split=0.2,
              early_stopping=True):

        """
        Trains a neural network to estimate either the likelihood ratio or, if method is 'sally' or 'sallino', the
        score.

        Parameters
        ----------
        method : {'alice', 'alice2', 'alices', 'alices2', 'carl', 'carl2', 'nde', 'rascal', 'rascal2', 'rolr', 'rolr2',
        'sally', 'sallino', 'scandal'}
            The inference method used:

             * For 'alice', 'alices', 'carl', 'nde', 'rascal', 'rolr', and 'scandal', the neural network models
             the likelihood ratio as a function of the observables `x` and the numerator hypothesis `theta0`, while
             the denominator hypothesis is kept at a fixed reference value. The score at `theta0` can also be evaluated.

             * For 'alice2', 'alices2', 'carl2', 'rascal2', and 'rolr2', the neural network models
             the likelihood ratio as a function of the observables `x`, the numerator hypothesis `theta0`, and the
             denominator hypothesis `theta1`. The score at `theta0` and `theta1` can also be evaluated.

             * For 'sally' and 'sallino', the neural networks models the score evaluated at some reference hypothesis.
             The likelihood ratio cannot be estimated directly from the neural network, but can be estimated in a second
             step through density estimation in the estimated score space.
            
        x_filename : str
            Path to an unweighted sample of observations, as saved by the `madminer.sampling.SampleAugmenter` functions.
            Required for all inference methods.
            
        y_filename : str or None, optional
            Path to an unweighted sample of class labels, as saved by the `madminer.sampling.SampleAugmenter` functions.
            Required for the 'alice', 'alice2', 'alices', 'alices2', 'carl', 'carl2', 'rascal', 'rascal2', 'rolr',
            and 'rolr2' methods. Default value: None.

        theta0_filename : str or None, optional
            Path to an unweighted sample of numerator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required for the 'alice', 'alice2', 'alices', 'alices2', 'carl', 'carl2', 'nde', 'rascal',
            'rascal2', 'rolr', 'rolr2', and 'scandal' methods. Default value: None.

        theta1_filename : str or None, optional
            Path to an unweighted sample of denominator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required for the 'alice2', 'alices2', 'carl2', 'rascal2', and 'rolr2' methods. Default value:
            None.

        r_xz_filename : str or None, optional
            Path to an unweighted sample of joint likelihood ratios, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required for the 'alice', 'alice2', 'alices', 'alices2', 'rascal', 'rascal2', 'rolr', and 'rolr2'
            methods. Default value: None.

        t_xz0_filename : str or None, optional
            Path to an unweighted sample of joint scores at theta0, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required for the 'alices', 'alices2', 'rascal', 'rascal2', 'sallino', 'sally', and 'scandal'
            methods. Default value: None.

        t_xz1_filename : str or None, optional
            Path to an unweighted sample of joint scores at theta1, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required for the 'rascal2' and 'alices2' methods. Default value: None.

        features : list of int or None, optional
            Indices of observables (features) that are used as input to the neural networks. If None, all observables
            are used. Default value: None.

        nde_type : {'maf', 'mafmog'}, optional
            If the method is 'nde' or 'scandal', nde_type determines the architecture used in the neural density
            estimator. Currently supported are 'maf' for a Masked Autoregressive Flow with a Gaussian base density, or
            'mafmog' for a Masked Autoregressive Flow with a mixture of Gaussian base densities. Default value: 'maf'.

        n_hidden : int, optional
            Number of hidden layers in the neural networks. If method is 'nde' or 'scandal', this refers to the number
            of hidden layers in each individual MADE layer. Default value: 100.
            
        activation : {'tanh', 'sigmoid', 'relu'}, optional
            Activation function. Default value: 'tanh'

        maf_n_mades : int, optional
            If method is 'nde' or 'scandal', this sets the number of MADE layers. Default value: 3.

        maf_batch_norm : bool, optional
            If method is 'nde' or 'scandal', switches batch normalization layers after each MADE layer on or off.
            Default: True.

        maf_batch_norm_alpha : float, optional
            If method is 'nde' or 'scandal' and maf_batch_norm is True, this sets the alpha parameter in the calculation
            of the running average of the mean and variance. Default value: 0.1.

        maf_mog_n_components : int, optional
            If method is 'nde' or 'scandal' and nde_type is 'mafmog', this sets the number of Gaussian base components.
            Default value: 10.

        alpha : float, optional
            Hyperparameter weighting the score error in the loss function of the 'alices', 'alices2', 'rascal',
            'rascal2', and 'scandal' methods.

        n_epochs : int, optional
            Number of epochs. Default value: 20.

        batch_size : int, optional
            Batch size. Default value: 128.

        initial_lr : float, optional
            Learning rate during the first epoch, after which it exponentially decays to final_lr. Default value:
            0.002.

        final_lr : float, optional
            Learning rate during the last epoch. Default value: 0.0001.

        validation_split : float or None, optional
            Fraction of samples used  for validation and early stopping (if early_stopping is True). If None, the entire
            sample is used for training and early stopping is deactivated. Default value: 0.2.

        early_stopping : bool, optional
            Activates early stopping based on the validation loss (only if validation_split is not None).

        Returns
        -------
            None

        """

        logging.info('Starting training')
        logging.info('  Method:                 %s', method)
        logging.info('  Training data: x at %s', x_filename)
        if theta0_filename is not None:
            logging.info('                 theta0 at %s', theta0_filename)
        if theta1_filename is not None:
            logging.info('                 theta1 at %s', theta1_filename)
        if y_filename is not None:
            logging.info('                 y at %s', y_filename)
        if r_xz_filename is not None:
            logging.info('                 r_xz at %s', r_xz_filename)
        if t_xz0_filename is not None:
            logging.info('                 t_xz (theta0) at  %s', t_xz0_filename)
        if t_xz1_filename is not None:
            logging.info('                 t_xz (theta1) at  %s', t_xz1_filename)
        if features is None:
            logging.info('  Features:               all')
        else:
            logging.info('  Features:               %s', features)
        logging.info('  Method:                 %s', method)
        if method in ['nde', 'scandal']:
            logging.info('  Neural density est.:    %s', nde_type)
        if method not in ['nde', 'scandal']:
            logging.info('  Hidden layers:          %s', n_hidden)
        if method in ['nde', 'scandal']:
            logging.info('  MAF, number MADEs:      %s', maf_n_mades)
            logging.info('  MAF, batch norm:        %s', maf_batch_norm)
            logging.info('  MAF, BN alpha:          %s', maf_batch_norm_alpha)
            logging.info('  MAF MoG, components:    %s', maf_mog_n_components)
        logging.info('  Activation function:    %s', activation)
        if method in ['cascal', 'cascal2', 'rascal', 'rascal2', 'scandal']:
            logging.info('  alpha:                  %s', alpha)
        logging.info('  Batch size:             %s', batch_size)
        logging.info('  Epochs:                 %s', n_epochs)
        logging.info('  Learning rate:          %s initially, decaying to %s', initial_lr, final_lr)
        logging.info('  Validation split:       %s', validation_split)
        logging.info('  Early stopping:         %s', early_stopping)

        # Load training data
        logging.info('Loading training data')

        theta0 = load_and_check(theta0_filename)
        theta1 = load_and_check(theta1_filename)
        x = load_and_check(x_filename)
        y = load_and_check(y_filename)
        r_xz = load_and_check(r_xz_filename)
        t_xz0 = load_and_check(t_xz0_filename)
        t_xz1 = load_and_check(t_xz1_filename)

        if y is not None:
            y = y.reshape((-1, 1))

        # Check necessary information is theere
        assert x is not None
        if method in ['carl', 'carl2', 'nde', 'scandal', 'rolr', 'alice', 'rascal', 'alices', 'rolr2', 'alice2',
                      'rascal2', 'alices2']:
            assert theta0 is not None
        if method in ['rolr', 'alice', 'rascal', 'alices', 'rolr2', 'alice2', 'rascal2', 'alices2']:
            assert r_xz is not None
        if method in ['carl', 'carl2', 'rolr', 'alice', 'rascal', 'alices', 'rolr2', 'alice2', 'rascal2', 'alices2']:
            assert y is not None
        if method in ['scandal', 'rascal', 'alices', 'rascal2', 'alices2', 'sally', 'sallino']:
            assert t_xz0 is not None
        if method in ['carl2', 'rolr2', 'alice2', 'rascal2', 'alices2']:
            assert theta1 is not None
        if method in ['rascal2', 'alices2']:
            assert t_xz1 is not None

        if method in ['nde', 'scandal']:
            assert nde_type in ['maf', 'maf_mog']

        # Infer dimensions of problem
        n_samples = x.shape[0]
        n_observables = x.shape[1]
        if theta0 is not None:
            n_parameters = theta0.shape[1]
        else:
            n_parameters = t_xz0.shape[1]

        logging.info('Found %s samples with %s parameters and %s observables', n_samples, n_parameters, n_observables)

        # Features
        if features is not None:
            x = x[:, features]
            logging.info('Only using %s of %s observables', x.shape[1], n_observables)
            n_observables = x.shape[1]

        # Save setup
        self.method = method
        self.n_observables = n_observables
        self.n_parameters = n_parameters
        self.n_hidden = n_hidden
        self.activation = activation
        self.maf_n_mades = maf_n_mades
        self.maf_batch_norm = maf_batch_norm
        self.maf_batch_norm_alpha = maf_batch_norm_alpha
        self.features = features

        # Create model
        logging.info('Creating model for method %s', method)
        if method in ['carl', 'rolr', 'rascal', 'alice', 'alices']:
            self.method_type = 'parameterized'
            self.model = ParameterizedRatioEstimator(
                n_observables=n_observables,
                n_parameters=n_parameters,
                n_hidden=n_hidden,
                activation=activation
            )
        elif method in ['carl2', 'rolr2', 'rascal2', 'alice2', 'alices2']:
            self.method_type = 'doubly_parameterized'
            self.model = DoublyParameterizedRatioEstimator(
                n_observables=n_observables,
                n_parameters=n_parameters,
                n_hidden=n_hidden,
                activation=activation
            )
        elif method in ['sally', 'sallino']:
            self.method_type = 'local_score'
            self.model = LocalScoreEstimator(
                n_observables=n_observables,
                n_parameters=n_parameters,
                n_hidden=n_hidden,
                activation=activation
            )
        elif method in ['nde', 'scandal']:
            self.method_type = 'nde'
            if nde_type == 'maf':
                self.model = ConditionalMaskedAutoregressiveFlow(
                    n_conditionals=n_parameters,
                    n_inputs=n_observables,
                    n_hiddens=n_hidden,
                    n_mades=maf_n_mades,
                    activation=activation,
                    batch_norm=maf_batch_norm,
                    alpha=maf_batch_norm_alpha
                )
            elif nde_type == 'maf_mog':
                self.model = ConditionalMixtureMaskedAutoregressiveFlow(
                    n_conditionals=n_parameters,
                    n_inputs=n_observables,
                    n_components=maf_mog_n_components,
                    n_hiddens=n_hidden,
                    n_mades=maf_n_mades,
                    activation=activation,
                    batch_norm=maf_batch_norm,
                    alpha=maf_batch_norm_alpha
                )
        else:
            raise NotImplementedError('Unknown method {}'.format(method))

        # Loss fn
        if method in ['carl', 'carl2']:
            loss_functions = [ratio_losses.standard_cross_entropy]
            loss_weights = [1.]
            loss_labels = ['xe']

        elif method in ['rolr', 'rolr2']:
            loss_functions = [ratio_losses.ratio_mse]
            loss_weights = [1.]
            loss_labels = ['mse_r']

        elif method == 'rascal':
            loss_functions = [ratio_losses.ratio_mse, ratio_losses.score_mse_num]
            loss_weights = [1., alpha]
            loss_labels = ['mse_r', 'mse_score']

        elif method == 'rascal2':
            loss_functions = [ratio_losses.ratio_mse, ratio_losses.score_mse]
            loss_weights = [1., alpha]
            loss_labels = ['mse_r', 'mse_score']

        elif method in ['alice', 'alice2']:
            loss_functions = [ratio_losses.augmented_cross_entropy]
            loss_weights = [1.]
            loss_labels = ['improved_xe']

        elif method == 'alices':
            loss_functions = [ratio_losses.augmented_cross_entropy, ratio_losses.score_mse_num]
            loss_weights = [1., alpha]
            loss_labels = ['improved_xe', 'mse_score']

        elif method == 'alices2':
            loss_functions = [ratio_losses.augmented_cross_entropy, ratio_losses.score_mse]
            loss_weights = [1., alpha]
            loss_labels = ['improved_xe', 'mse_score']

        elif method in ['sally', 'sallino']:
            loss_functions = [ratio_losses.local_score_mse]
            loss_weights = [1.]
            loss_labels = ['mse_score']

        elif method == 'nde':
            loss_functions = [flow_losses.negative_log_likelihood]
            loss_weights = [1.]
            loss_labels = ['nll']

        elif method == 'scandal':
            loss_functions = [flow_losses.negative_log_likelihood, flow_losses.score_mse]
            loss_weights = [1., alpha]
            loss_labels = ['nll', 'mse_score']

        else:
            raise NotImplementedError('Unknown method {}'.format(method))

        # Train model
        logging.info('Training model')

        if method in ['sally', 'sallino']:
            train_local_score_model(
                model=self.model,
                loss_functions=loss_functions,
                loss_weights=loss_weights,
                loss_labels=loss_labels,
                xs=x,
                t_xzs=t_xz0,
                batch_size=batch_size,
                n_epochs=n_epochs,
                initial_learning_rate=initial_lr,
                final_learning_rate=final_lr,
                validation_split=validation_split,
                early_stopping=early_stopping
            )
        elif method in ['nde', 'scandal']:
            train_flow_model(
                model=self.model,
                loss_functions=loss_functions,
                loss_weights=loss_weights,
                loss_labels=loss_labels,
                xs=x,
                t_xz0s=t_xz0,
                batch_size=batch_size,
                n_epochs=n_epochs,
                initial_learning_rate=initial_lr,
                final_learning_rate=final_lr,
                validation_split=validation_split,
                early_stopping=early_stopping
            )
        else:
            train_ratio_model(
                model=self.model,
                method_type=self.method_type,
                loss_functions=loss_functions,
                loss_weights=loss_weights,
                loss_labels=loss_labels,
                theta0s=theta0,
                theta1s=theta1,
                xs=x,
                ys=y,
                r_xzs=r_xz,
                t_xz0s=t_xz0,
                t_xz1s=t_xz1,
                batch_size=batch_size,
                n_epochs=n_epochs,
                initial_learning_rate=initial_lr,
                final_learning_rate=final_lr,
                validation_split=validation_split,
                early_stopping=early_stopping
            )

    def evaluate(self,
                 x_filename,
                 theta0_filename=None,
                 theta1_filename=None,
                 test_all_combinations=True,
                 evaluate_score=False):

        """
        Evaluates a trained estimator of the likelihood ratio (or, if method is 'sally' or 'sallino', the score).

        Parameters
        ----------
        x_filename : str
            Path to an unweighted sample of observations, as saved by the `madminer.sampling.SampleAugmenter` functions.
            
        theta0_filename : str or None, optional
            Path to an unweighted sample of numerator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required if the estimator was trained with the 'alice', 'alice2', 'alices', 'alices2', 'carl',
            'carl2', 'nde', 'rascal', 'rascal2', 'rolr', 'rolr2', or 'scandal' method. Default value: None.

        theta1_filename : str or None, optional
            Path to an unweighted sample of denominator parameters, as saved by the `madminer.sampling.SampleAugmenter`
            functions. Required if the estimator was trained with the 'alice2', 'alices2', 'carl2', 'rascal2', or
            'rolr2' method. Default value: None.

        test_all_combinations : bool, optional
            If method is not 'sally' and not 'sallino': If False, the number of samples in the observable and theta
            files has to match, and the likelihood ratio is evaluated only for the combinations
            `r(x_i | theta0_i, theta1_i)`. If True, `r(x_i | theta0_j, theta1_j)` for all pairwise combinations `i, j`
            are evaluated. Default value: True.

        evaluate_score : bool, optional
            If method is not 'sally' and not 'sallino', this sets whether in addition to the likelihood ratio the score
            is evaluated. Default value: False.

        Returns
        -------
        sally_estimated_score : ndarray
            Only returned if the network was trained with `method='sally'` or `method='sallino'`. In this case, an
            array of the estimator for `t(x_i | theta_ref)` is returned for all events `i`.

        log_likelihood_ratio : ndarray
            Only returned if the network was trained with neither `method='sally'` nor `method='sallino'`. The estimated
            likelihood ratio. If test_all_combinations is True, the result has shape `(n_thetas, n_x)`. Otherwise, it
            has shape `(n_samples,)`.

        score_theta0 : ndarray or None
            Only returned if the network was trained with neither `method='sally'` nor `method='sallino'`. None if
            evaluate_score is False. Otherwise the derived estimated score at `theta0`. If test_all_combinations is
            True, the result has shape `(n_thetas, n_x, n_parameters)`. Otherwise, it has shape
            `(n_samples, n_parameters)`.

        score_theta1 : ndarray or None
            Only returned if the network was trained with neither `method='sally'` nor `method='sallino'`. None if
            evaluate_score is False, or the network was trained with any method other than 'alice2', 'alices2', 'carl2',
            'rascal2', or 'rolr2'. Otherwise the derived estimated score at `theta1`. If test_all_combinations is
            True, the result has shape `(n_thetas, n_x, n_parameters)`. Otherwise, it has shape
            `(n_samples, n_parameters)`.

        """

        if self.model is None:
            raise ValueError('No model -- train or load model before evaluating it!')

        # Load training data
        logging.info('Loading evaluation data')

        theta0s = load_and_check(theta0_filename)
        theta1s = load_and_check(theta1_filename)
        xs = load_and_check(x_filename)

        # Restrict featuers
        if self.features is not None:
            xs = xs[:, self.features]

        # SALLY evaluation
        if self.method in ['sally', 'sallino']:
            logging.info('Starting score evaluation')

            all_t_hat = evaluate_local_score_model(
                model=self.model,
                xs=xs
            )

            return all_t_hat

        # Balance thetas
        if theta1s is None and theta0s is not None:
            theta1s = [None for _ in theta0s]
        elif theta1s is not None and theta0s is not None:
            if len(theta1s) > len(theta0s):
                theta0s = [theta0s[i % len(theta0s)] for i in range(len(theta1s))]
            elif len(theta1s) < len(theta0s):
                theta1s = [theta1s[i % len(theta1s)] for i in range(len(theta0s))]

        # Evaluation for all other methods
        all_log_r_hat = []
        all_t_hat0 = []
        all_t_hat1 = []

        if test_all_combinations:
            logging.info('Starting ratio evaluation for all combinations')

            for i, (theta0, theta1) in enumerate(zip(theta0s, theta1s)):
                logging.debug('Starting ratio evaluation for thetas %s / %s: %s vs %s',
                              i + 1, len(theta0s), theta0, theta1)

                if self.method in ['nde', 'scandal']:
                    _, log_r_hat, t_hat0 = evaluate_flow_model(
                        model=self.model,
                        theta0s=[theta0],
                        xs=xs,
                        evaluate_score=evaluate_score
                    )
                    t_hat1 = None

                else:
                    _, log_r_hat, t_hat0, t_hat1 = evaluate_ratio_model(
                        model=self.model,
                        method_type=self.method_type,
                        theta0s=[theta0],
                        theta1s=[theta1] if theta1 is not None else None,
                        xs=xs,
                        evaluate_score=evaluate_score
                    )

                all_log_r_hat.append(log_r_hat)
                all_t_hat0.append(t_hat0)
                all_t_hat1.append(t_hat1)

            all_log_r_hat = np.array(all_log_r_hat)
            all_t_hat0 = np.array(all_t_hat0)
            all_t_hat1 = np.array(all_t_hat1)

        else:
            logging.info('Starting ratio evaluation')

            if self.method in ['nde', 'scandal']:
                _, all_log_r_hat, t_hat0 = evaluate_flow_model(
                    model=self.model,
                    theta0s=theta0s,
                    xs=xs,
                    evaluate_score=evaluate_score
                )
                all_t_hat1 = None

            else:
                _, all_log_r_hat, all_t_hat0, all_t_hat1 = evaluate_ratio_model(
                    model=self.model,
                    method_type=self.method_type,
                    theta0s=theta0s,
                    theta1s=None if None in theta1s else theta1s,
                    xs=xs,
                    evaluate_score=evaluate_score
                )

        logging.info('Evaluation done')

        return all_log_r_hat, all_t_hat0, all_t_hat1

    def calculate_fisher_information(self,
                                     x_filename,
                                     n_events=1):

        """
        Calculates the expected Fisher information matrix based on the kinematic information in a given number of
        events. Currently only supported for estimators trained with `method='sally'` or `method='sallino'`.

        Parameters
        ----------
        x_filename : str
            Path to an unweighted sample of observations, as saved by the `madminer.sampling.SampleAugmenter` functions.
            Note that this sample has to be sample from the reference parameter where the score is estimated with the
            SALLY / SALLINO estimator!
            
        n_events : int, optional
            Number of events for which the kinematic Fisher information should be calculated. Default value: 1.

        Returns
        -------
        fisher_information : ndarray
            Expected kinematic Fisher information matrix with shape `(n_parameters, n_parameters)`.

        """

        if self.model is None:
            raise ValueError('No model -- train or load model before evaluating it!')

        # Load training data
        logging.info('Loading evaluation data')
        xs = load_and_check(x_filename)
        n_samples = xs.shape[0]

        # Restrict featuers
        if self.features is not None:
            xs = xs[:, self.features]

        # Estimate scores
        if self.method in ['sally', 'sallino']:
            logging.info('Starting score evaluation')

            t_hats = evaluate_local_score_model(
                model=self.model,
                xs=xs
            )
        else:
            raise NotImplementedError('Fisher information calculation only implemented for SALLY estimators')

        # Calculate Fisher information
        n_parameters = t_hats.shape[1]
        fisher_information = np.zeros((n_parameters, n_parameters))
        for t_hat in t_hats:
            fisher_information += np.outer(t_hat, t_hat)
        fisher_information = float(n_events) / float(n_samples) * fisher_information

        # Calculate expected score
        expected_score = np.mean(t_hats, axis=0)
        logging.info('Expected score (should be close to zero): %s', expected_score)

        return fisher_information

    def save(self,
             filename):

        """
        Saves the trained model to two files: a JSON file with the settings, as well as a pickled pyTorch state dict
        file.

        Parameters
        ----------
        filename : str
            Path to the files. '_settings.json' and '_state_dict.pl' will be added.

        Returns
        -------
            None

        """

        if self.model is None:
            raise ValueError('No model -- train or load model before saving!')

        # Check paths
        create_missing_folders([os.path.dirname(filename)])

        # Save settings
        logging.info('Saving settings to %s_settings.json', filename)

        settings = {'method': self.method,
                    'method_type': self.method_type,
                    'n_observables': self.n_observables,
                    'n_parameters': self.n_parameters,
                    'n_hidden': list(self.n_hidden),
                    'activation': self.activation,
                    'features': self.features}

        with open(filename + '_settings.json', 'w') as f:
            json.dump(settings, f)

        # Save state dict
        logging.info('Saving state dictionary to %s_state_dict.pt', filename)
        torch.save(self.model.state_dict(), filename + '_state_dict.pt')

    def load(self,
             filename):

        """
        Loads a trained model from files.

        Parameters
        ----------
        filename : str
            Path to the files. '_settings.json' and '_state_dict.pl' will be added.

        Returns
        -------
            None

        """

        # Load settings
        logging.info('Loading settings from %s_settings.json', filename)

        with open(filename + '_settings.json', 'r') as f:
            settings = json.load(f)

        self.method = settings['method']
        self.method_type = settings['method_type']
        self.n_observables = int(settings['n_observables'])
        self.n_parameters = int(settings['n_parameters'])
        self.n_hidden = tuple([int(item) for item in settings['n_hidden']])
        self.activation = str(settings['activation'])
        self.features = settings['features']
        if self.features == 'None':
            self.features = None
        if self.features is not None:
            self.features = list([int(item) for item in self.features])

        logging.info('  Found method %s, %s observables, %s parameters, %s hidden layers, %s activation function, '
                     'features %s',
                     self.method, self.n_observables, self.n_parameters, self.n_hidden, self.activation, self.features)

        # Create model
        if self.method in ['carl', 'rolr', 'rascal', 'alice', 'alices']:
            assert self.method_type == 'parameterized'
            self.model = ParameterizedRatioEstimator(
                n_observables=self.n_observables,
                n_parameters=self.n_parameters,
                n_hidden=self.n_hidden,
                activation=self.activation
            )
        elif self.method in ['carl2', 'rolr2', 'rascal2', 'alice2', 'alices2']:
            assert self.method_type == 'doubly_parameterized'
            self.model = DoublyParameterizedRatioEstimator(
                n_observables=self.n_observables,
                n_parameters=self.n_parameters,
                n_hidden=self.n_hidden,
                activation=self.activation
            )
        elif self.method in ['sally', 'sallino']:
            assert self.method_type == 'local_score'
            self.model = LocalScoreEstimator(
                n_observables=self.n_observables,
                n_parameters=self.n_parameters,
                n_hidden=self.n_hidden,
                activation=self.activation
            )
        else:
            raise NotImplementedError('Unknown method {}'.format(self.method))

        # Load state dict
        logging.info('Loading state dictionary from %s_state_dict.pt', filename)
        self.model.load_state_dict(torch.load(filename + '_state_dict.pt'))


class EnsembleForge:

    def __init__(self, estimators=None):
        if estimators is None:
            estimators = []

        self.estimators = estimators
        self.n_estimators = len(self.estimators)
        self.expectations = None

        # Consistency checks
        for estimator in self.estimators:
            assert isinstance(estimator, MLForge), 'Estimator is no MLForge instance!'

        self._check_consistency()

    def train_one(self, i, **kwargs):
        self._check_consistency(kwargs)

        self.estimators[i].train(**kwargs)

    def train_all_same(self, **kwargs):
        self._check_consistency(kwargs)

        for estimator in self.estimators:
            estimator.train(**kwargs)

    def train_all_differently(self, **kwargs):
        for key, value in six.iteritems(kwargs):
            if not isinstance(value, list):
                kwargs[key] = [value for _ in range(self.n_estimators)]

            assert len(key) == self.n_estimators, 'Keyword {} has wrong length {}'.format(key, len(value))

        self._check_consistency(kwargs)

        for i, estimator in enumerate(self.estimators):
            kwargs_this_estimator = {}
            for key, value in six.iteritems(kwargs):
                kwargs_this_estimator[key] = value[i]

            estimator.train(**kwargs_this_estimator)

    def calculate_expectation(self,
                              x_filename,
                              theta0_filename=None,
                              theta1_filename=None):

        self.expectations = []
        method_type = self._check_consistency()

        for estimator in self.estimators:
            # Calculate expected score / ratio
            if method_type == 'local_score':
                prediction = estimator.evaluate(x_filename, theta0_filename, theta1_filename)
            else:
                raise NotImplementedError('Expectation calculation currently only implemented for SALLY and SALLINO!')

            self.expectations.append(
                np.mean(prediction, axis=0)
            )

        self.expectations = np.array(self.expectations)

        return self.expectations

    def evaluate(self,
                 x_filename,
                 theta0_filename=None,
                 theta1_filename=None,
                 test_all_combinations=True,
                 evaluate_score=False,
                 vote_expectation_weight=None,
                 return_individual_predictions=False):

        # Calculate weights of each estimator in vote
        if self.expectations is None or vote_expectation_weight is None:
            weights = np.ones(self.n_estimators)
        else:
            weights = np.exp(-vote_expectation_weight * self.expectations)

        weights /= np.sum(weights)

        # Calculate estimator predictions
        predictions = []
        for estimator in self.estimators:
            predictions.append(estimator.evaluate(
                x_filename,
                theta0_filename,
                theta1_filename,
                test_all_combinations,
                evaluate_score
            ))
        predictions = np.array(predictions)

        # Calculate weighted mean
        mean = np.average(predictions, axis=0, weights=weights)

        # Calculate weighted variance
        if self.n_estimators > 1:
            variance = np.average((predictions - mean) ** 2, axis=0, weights=weights)

            # Correct bias, see https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights.
            bias = 1. - np.sum(weights ** 2) / np.sum(weights) ** 2
            variance /= bias
        else:
            logging.warning('Only one estimator, no meaningful variance calculation!')
            variance = np.zeros_like(mean)

        std = np.sqrt(variance)

        if return_individual_predictions:
            return mean, std, weights, predictions

        return mean, std

    def calculate_fisher_information(self,
                                     x_filename,
                                     n_events=1,
                                     vote_expectation_weight=None,
                                     return_individual_predictions=False):

        # Calculate weights of each estimator in vote
        if self.expectations is None or vote_expectation_weight is None:
            weights = np.ones(self.n_estimators)
        else:
            weights = np.exp(-vote_expectation_weight * self.expectations)

        weights /= np.sum(weights)

        # Calculate estimator predictions
        predictions = []
        for estimator in self.estimators:
            predictions.append(estimator.calculate_fisher_information(
                x_filename=x_filename,
                n_events=n_events
            ))
        predictions = np.array(predictions)

        # Calculate weighted mean
        mean = np.average(predictions, axis=0, weights=weights)

        # Calculate weighted variance
        if self.n_estimators > 1:
            variance = np.average((predictions - mean) ** 2, axis=0, weights=weights)
            variance *= float(self.n_estimators) / float(self.n_estimators - 1.)  # Unbiased estimator of pop. var.
        else:
            logging.warning('Only one estimator, no meaningful variance calculation!')
            variance = np.zeros_like(mean)

        std = np.sqrt(variance)

        if return_individual_predictions:
            return mean, std, weights, predictions

        return mean, std

    def save(self, folder):
        # Check paths
        create_missing_folders([folder])

        # Save ensemble settings
        logging.info('Saving ensemble setup to %/ensemble.json', folder)

        settings = {'n_estimators': self.n_estimators,
                    'expectations': 'None' if self.expectations is None else list(self.expectations)}

        with open(folder + '/ensemble.json', 'w') as f:
            json.dump(settings, f)

        # Save estimators
        for i, estimator in enumerate(self.estimators):
            estimator.save(folder + '/estimator_' + str(i))

    def load(self, folder):
        # Load ensemble settings
        logging.info('Loading ensemble setup from %/ensemble.json', folder)

        with open(folder + '/ensemble.json', 'r') as f:
            settings = json.load(f)

        self.n_estimators = settings['n_estimators']
        self.expectations = settings['expectations']
        if self.expectations == 'None':
            self.expectations = None
        if self.expectations is not None:
            self.expectations = np.array(self.expectations)

        logging.info('  Found ensemble with %s estimators and expectations %s',
                     self.n_estimators, self.expectations)

        # Load estimators
        self.estimators = []
        for i in range(self.n_estimators):
            estimator = MLForge()
            estimator.load(folder + '/estimator_' + str(i))
            self.estimators.append(estimator)

    def _check_consistency(self, keywords=None):
        # Accumulate methods of all estimators
        methods = [estimator.method for estimator in self.estimators]

        if keywords is not None:
            keyword_method = keywords.get("method", None)
            if isinstance(keyword_method, list):
                methods += keyword_method
            else:
                methods.append(keyword_method)

        # Check consistency
        method_type = None
        for method in methods:
            if method in ['sally', 'sallino']:
                this_method_type = 'local_score'
            elif method in ['carl', 'rolr', 'rascal', 'alice', 'alices', 'nde', 'scandal']:
                # this_method_type = 'parameterized'
                raise NotImplementedError('For now, ensemble methods are only implemented for SALLY and SALLINO.')
            elif method in ['carl2', 'rolr2', 'rascal2', 'alice2', 'alices2']:
                # this_method_type = 'doubly_parameterized'
                raise NotImplementedError('For now, ensemble methods are only implemented for SALLY and SALLINO.')
            elif method is None:
                continue
            else:
                raise RuntimeError('Unknown method %s', method)

            if method_type is None:
                method_type = this_method_type

            if method_type != this_method_type:
                raise RuntimeError('Ensemble with inconsistent estimator methods! All methods have to be either'
                                   ' single-parameterized ratio estimators, doubly parameterized ratio estimators,'
                                   ' or local score estimators. Found methods ' + ', '.join(methods) + '.')

        # Return method type of ensemble
        return method_type
