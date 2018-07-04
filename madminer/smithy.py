from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import collections

from madminer.tools.h5_interface import load_madminer_settings, madminer_event_loader
from madminer.tools.analysis import get_theta_value, get_theta_benchmark_matrix, get_dtheta_benchmark_matrix
from madminer.tools.analysis import extract_augmented_data
from madminer.tools.morphing import Morpher


def constant_benchmark_theta(benchmark_name):
    return 'benchmark', benchmark_name


def multiple_benchmark_thetas(benchmark_names):
    return 'benchmarks', benchmark_names


def constant_morphing_theta(theta):
    return 'theta', theta


def multiple_morphing_thetas(thetas):
    return 'thetas', thetas


def random_morphing_thetas(n_thetas, prior):
    return 'random', n_thetas, prior


class Smithy:

    def __init__(self, filename, disable_morphing=False):
        self.madminer_filename = filename

        # Load data
        (self.parameters, self.benchmarks, self.morphing_components, self.morphing_matrix,
         self.observables) = load_madminer_settings(filename)

        # Morphing
        self.morpher = None
        if self.morphing_matrix is not None and self.morphing_components is not None and not disable_morphing:
            self.morpher = Morpher(self.parameters)
            self.morpher.set_components(self.morphing_components)
            self.morpher.set_basis(self.benchmarks, self.morphing_matrix)

    def extract_samples_train_onlysamples(self,
                                          theta,
                                          n_samples,
                                          folder,
                                          filename,
                                          test_split=0.3):
        """
        Extracts training samples for histograms and ABC

        Sampling: according to theta. theta can be fixed or varying.
        Data: theta, x
        """
        raise NotImplementedError

    def extract_samples_train_local(self,
                                    theta,
                                    n_samples,
                                    folder,
                                    filename,
                                    test_split=0.3):
        """
        Extracts training samples for SALLY and SALLINO

        Sampling: according to theta.
        Data: theta, x, t(x,z)
        """
        raise NotImplementedError

    def extract_samples_train_ratio(self,
                                    theta0,
                                    theta1,
                                    n_samples,
                                    folder,
                                    filename,
                                    test_split=0.3):
        """
        Extracts training samples for CARL, ROLR, CASCAL, RASCAL

        Sampling: 50% according to theta0, 50% according to theta1. theta0 can be fixed or varying.
        Data: theta0, theta1, x, y, r(x,z), t(x,z)
        """
        raise NotImplementedError

    def extract_samples_test(self,
                             theta,
                             n_samples,
                             folder,
                             filename,
                             test_split=0.3):
        """
        Extracts evaluation samples for all methods

        Sampling: according to theta
        Data: x
        """
        raise NotImplementedError

    def extract_sample(self,
                       theta_sampling_types,
                       theta_sampling_values,
                       n_samples_per_theta,
                       augmented_data_definitions,
                       start_event=0,
                       end_event=None):
        """
        Low-level function for the extraction of information from the event samples.

        :param theta_sampling_types: list, each entry is either 'benchmark' or 'morphing'
        :param theta_sampling_values: list, each entry is int and labels the benchmark index (if the corresponding
                                      theta_sampling_types entry is 'benchmark') or a numpy array with the theta values
                                      (of the corresponding theta_sampling_types entry is 'morphing')
        :param n_samples_per_theta:
        :param augmented_data_definitions:
        :param start_event:
        :param end_event:
        :return:
        """

        # Calculate total xsecs
        xsecs_benchmarks = None
        n_observables = 0

        for obs, weights in madminer_event_loader(self.madminer_filename, start=start_event, end=end_event):
            if xsecs_benchmarks is None:
                xsecs_benchmarks = np.sum(weights, axis=0)
            else:
                xsecs_benchmarks += np.sum(weights, axis=0)

            n_observables = obs.shape[1]

        # Consistency checks
        n_benchmarks = xsecs_benchmarks.shape[0]

        if n_benchmarks != len(self.benchmarks) and self.morphing_matrix is None:
            raise ValueError('Inconsistent numbers of benchmarks: {} in observations,'
                             '{} in benchmark list'.format(n_benchmarks, len(self.benchmarks)))
        elif n_benchmarks != len(self.benchmarks) or n_benchmarks != self.morphing_matrix.shape[0]:
            raise ValueError('Inconsistent numbers of benchmarks: {} in observations, {} in benchmark list, '
                             '{} in morphing matrix'.format(n_benchmarks, len(self.benchmarks),
                                                            self.morphing_matrix.shape[0]))

        if n_observables != len(self.observables):
            raise ValueError('Inconsistent numbers of observables: {} in observations,'
                             '{} in observable list'.format(n_observables, len(self.observables)))

        # Prepare augmented data
        n_augmented_data = len(augmented_data_definitions)
        augmented_data_types = []
        augmented_data_theta_matrices_num = []
        augmented_data_theta_matrices_den = []

        for augmented_data_definition in augmented_data_definitions:
            augmented_data_types.append(augmented_data_definition[0])

            if augmented_data_types[-1] == 'ratio':
                augmented_data_theta_matrices_num.append(
                    get_theta_benchmark_matrix(
                        augmented_data_definition[1],
                        augmented_data_definition[2],

                        n_benchmarks,
                        self.morpher
                    )
                )
                augmented_data_theta_matrices_den.append(
                    get_theta_benchmark_matrix(
                        augmented_data_definition[3],
                        augmented_data_definition[4],
                        n_benchmarks,
                        self.morpher
                    )
                )

            elif augmented_data_types[-1] == 'score':
                augmented_data_theta_matrices_num.append(
                    get_dtheta_benchmark_matrix(
                        augmented_data_definition[1],
                        augmented_data_definition[2],
                        n_benchmarks,
                        self.morpher
                    )
                )
                augmented_data_theta_matrices_den.append(
                    get_theta_benchmark_matrix(
                        augmented_data_definition[1],
                        augmented_data_definition[2],
                        n_benchmarks,
                        self.morpher
                    )
                )

        # Samples per theta
        if not isinstance(n_samples_per_theta, collections.Iterable) or len(n_samples_per_theta) == 1:
            n_samples_per_theta = [n_samples_per_theta] * len(theta_sampling_types)

        # Prepare output
        all_theta = []
        all_x = []
        all_augmented_data = []

        # Loop over thetas
        for theta_type, theta_value, n_samples in zip(theta_sampling_types, theta_sampling_values, n_samples_per_theta):

            # Prepare output
            samples_done = np.zeros(n_samples, dtype=np.bool)
            samples_x = np.zeros((n_samples, n_observables))
            samples_augmented_data = np.zeros((n_samples, n_augmented_data))

            # Sampling theta
            theta_values = get_theta_value(theta_type, theta_value, self.benchmarks)
            theta_values = np.broadcast_to(theta_values, (n_samples, theta_values.size))
            theta_matrix = get_theta_benchmark_matrix(theta_type, theta_value, n_benchmarks, self.morpher)

            # Total xsec for this theta
            xsec_theta = theta_matrix.dot(xsecs_benchmarks)

            # Draw random numbers in [0, 1]
            u = np.random.rand(n_samples)

            cumulative_p = [0.]

            for x_batch, weights_benchmarks_batch in madminer_event_loader(self.madminer_filename,
                                                                           start=start_event,
                                                                           end=end_event):
                # Evaluate cumulative p(x | theta)
                weights_theta = theta_matrix.dot(weights_benchmarks_batch.T)  # Shape (n_events_in_batch,)
                p_theta = weights_theta / xsec_theta
                cumulative_p = cumulative_p[-1] + np.cumsum(p_theta)

                # Check what we've found
                indices = np.searchsorted(cumulative_p, u, side='left').flatten()

                found_now = (np.invert(samples_done) & (indices < len(cumulative_p)))

                # Save x
                samples_x[found_now] = x_batch[indices[found_now]]

                # Extract augmented data
                relevant_augmented_data = extract_augmented_data(
                    augmented_data_types,
                    augmented_data_theta_matrices_num,
                    augmented_data_theta_matrices_den,
                    weights_benchmarks_batch[indices[found_now], :],
                    xsecs_benchmarks
                )

                samples_augmented_data[found_now] = relevant_augmented_data

                samples_done[found_now] = True

                if np.all(samples_done):
                    break

            # Check that we got 'em all
            if not np.all(samples_done):
                raise ValueError('{} / {} samples not found, u = {}', np.sum(np.invert(samples_done)),
                                 samples_done.size, u[np.invert(samples_done)])

            all_x.append(samples_x)
            all_augmented_data.append(samples_augmented_data)
            all_theta.append(theta_values)

        return all_theta, all_x, all_augmented_data
