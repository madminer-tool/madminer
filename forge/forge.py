from __future__ import absolute_import, division, print_function

import logging
import os
import json
import numpy as np

import torch

from forge.ml import losses
from forge.ml.models import ParameterizedRatioEstimator
from forge.ml.trainer import train_model, evaluate_model
from forge.ml.utils import create_missing_folders, load_and_check, general_init


class Forge:

    def __init__(self, debug=False):
        general_init(debug=debug)

        self.model = None
        self.method = None
        self.n_observables = None
        self.n_parameters = None
        self.n_hidden = None
        self.activation = None

    def train(self,
              method,
              theta_filename,
              x_filename,
              y_filename,
              r_xz_filename=None,
              t_xz_filename=None,
              n_hidden=(100, 100, 100),
              activation='tanh',
              alpha=1.,
              n_epochs=20,
              batch_size=64,
              initial_lr=0.001,
              final_lr=0.0001,
              validation_split=0.2,
              early_stopping=True):

        """ Trains likelihood ratio estimator """

        logging.info('Starting training')
        logging.info('  Method:                 %s', method)
        logging.info('  Training data: theta at %s', theta_filename)
        logging.info('                 x at     %s', x_filename)
        logging.info('                 y at     %s', y_filename)
        if r_xz_filename is not None:
            logging.info('                 r_xz at  %s', r_xz_filename)
        if t_xz_filename is not None:
            logging.info('                 t_xz at  %s', t_xz_filename)
        logging.info('  Method:                 %s', method)
        logging.info('  Hidden layers:          %s', n_hidden)
        logging.info('  Activation function:    %s', activation)
        logging.info('  alpha:                  %s', alpha)
        logging.info('  Batch size:             %s', batch_size)
        logging.info('  Epochs:                 %s', n_epochs)
        logging.info('  Learning rate:          %s initially, decaying to %s', initial_lr, final_lr)
        logging.info('  Early stopping:         %s', early_stopping)

        # Load training data
        logging.info('Loading training data')

        theta = load_and_check(theta_filename)
        x = load_and_check(x_filename)
        y = load_and_check(y_filename).reshape((-1, 1))
        r_xz = load_and_check(r_xz_filename)
        t_xz = load_and_check(t_xz_filename)

        n_samples = theta.shape[0]
        n_parameters = theta.shape[1]
        n_observables = x.shape[1]

        logging.info('Found %s samples with %s parameters and %s observables', n_samples, n_parameters, n_observables)

        # Save setup
        self.method = method
        self.n_observables = n_observables
        self.n_parameters = n_parameters
        self.n_hidden = n_hidden
        self.activation = activation

        # Create model
        logging.info('Creating model for method %s', method)
        if method in ['rolr', 'rascal', 'alice', 'alices']:
            self.model = ParameterizedRatioEstimator(
                n_observables=n_observables,
                n_parameters=n_parameters,
                n_hidden=n_hidden,
                activation=activation
            )
        else:
            raise NotImplementedError('Unknown method {}'.format(method))

        # Loss fn
        if method == 'rolr':
            loss_functions = [losses.ratio_mse]
            loss_weights = [1.]
            loss_labels = ['mse_r']

        elif method == 'rascal':
            loss_functions = [losses.ratio_mse, losses.score_mse_num]
            loss_weights = [1., alpha]
            loss_labels = ['mse_r', 'mse_score']

        elif method == 'alice':
            loss_functions = [losses.augmented_cross_entropy]
            loss_weights = [1.]
            loss_labels = ['improved_xe']

        elif method == 'alices':
            loss_functions = [losses.augmented_cross_entropy, losses.score_mse_num]
            loss_weights = [1., alpha]
            loss_labels = ['improved_xe', 'mse_score']

        else:
            raise NotImplementedError('Unknown method {}'.format(method))

        # Train model
        logging.info('Training model')

        train_model(
            model=self.model,
            loss_functions=loss_functions,
            loss_weights=loss_weights,
            loss_labels=loss_labels,
            thetas=theta,
            xs=x,
            ys=y,
            r_xzs=r_xz,
            t_xzs=t_xz,
            batch_size=batch_size,
            n_epochs=n_epochs,
            initial_learning_rate=initial_lr,
            final_learning_rate=final_lr,
            validation_split=validation_split,
            early_stopping=early_stopping
        )

    def evaluate(self,
                 theta_filename,
                 x_filename):

        """ Predicts log likelihood ratio for all combinations of theta and x """

        if self.model is None:
            raise ValueError('No model -- train or load model before evaluating it!')

        logging.info('Starting evaluation')

        # Load training data
        logging.info('Loading training data')

        thetas = load_and_check(theta_filename)
        xs = load_and_check(x_filename)

        # Loop over thetas
        all_log_r_hat = []
        all_t_hat = []

        for i, theta in enumerate(thetas):
            logging.debug('Starting evaluation for theta %s / %s, %s',
                          i + 1, len(thetas), theta)

            _, log_r_hat, t_hat = evaluate_model(
                model=self.model,
                thetas=[theta],
                xs=xs
            )

            all_log_r_hat.append(log_r_hat)
            all_t_hat.append(t_hat)

        # Return
        all_log_r_hat = np.array(all_log_r_hat)
        all_t_hat = np.array(all_t_hat)

        return all_log_r_hat, all_t_hat

    def save(self,
             filename):

        """ Saves model state dict """

        if self.model is None:
            raise ValueError('No model -- train or load model before saving!')

        # Check paths
        create_missing_folders([os.path.dirname(filename)])

        # Save settings
        logging.info('Saving settings to %s_settings.json', filename)

        settings = {'method': self.method,
                    'n_observables': self.n_observables,
                    'n_parameters': self.n_parameters,
                    'n_hidden': list(self.n_hidden),
                    'activation': self.activation}

        with open(filename + '_settings.json', 'w') as f:
            json.dump(settings, f)

        # Save state dict
        logging.info('Saving state dictionary to %s_state_dict.pt', filename)
        torch.save(self.model.state_dict(), filename + '_state_dict.pt')

    def load(self,
             filename):

        """ Loads model state dict from file """

        # Load settings
        logging.info('Loading settings from %s_settings.json', filename)

        with open(filename + '_settings.json', 'r') as f:
            settings = json.load(f)

        self.method = settings['method']
        self.n_observables = int(settings['n_observables'])
        self.n_parameters = int(settings['n_parameters'])
        self.n_hidden = tuple([int(item) for item in settings['n_hidden']])
        self.activation = str(settings['activation'])

        # Create model
        if self.method in ['rolr', 'rascal', 'alice', 'alices']:
            self.model = ParameterizedRatioEstimator(
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
