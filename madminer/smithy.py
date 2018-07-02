from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from madminer.tools.h5_interface import load_madminer_settings, madminer_event_loader
from madminer.tools.analysis import get_theta_benchmark_matrix, get_dtheta_benchmark_matrix, extract_augmented_data


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

    def __init__(self, filename):
        self.madminer_filename = filename

        # Load data
        (self.parameters, self.benchmarks, self.morphing_components, self.morphing_matrix,
         self.observables) = load_madminer_settings(filename)

        # Normalize xsecs of benchmarks
        # self.total_xsecs = calculate_total_xsecs(self.madminer_filename)

    def extract_samples_local(self,
                              theta,
                              n_samples_train,
                              n_samples_test,
                              folder,
                              filename,
                              test_split):
        """
        Extracts samples for SALLY and SALLINO

        Sampling: according to theta.
        Data: theta, x, t(x,z)
        """
        raise NotImplementedError

    def extract_samples_ratio(self,
                              theta0,
                              theta1,
                              n_samples_train,
                              n_samples_test,
                              folder,
                              filename):
        """
        Extracts samples for CARL, ROLR, CASCAL, RASCAL

        Sampling: 50% according to theta0, 50% according to theta1. theta0 can be fixed or varying, theta1 can be
        Data: theta0, theta1, x, y, r(x,z), t(x,z)
        """
        raise NotImplementedError

    def extract_sample(self,
                       thetas,
                       n_samples_per_theta,
                       augmented_data_definitions,
                       start_event=0,
                       end_event=None):

        # Calculate total xsecs
        xsecs_benchmarks = None
        n_observables = 0

        for obs, weights in madminer_event_loader(self.madminer_filename, start=start_event, end=end_event):
            if xsecs_benchmarks is None:
                xsecs_benchmarks = weights
            else:
                xsecs_benchmarks += weights

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
                    get_theta_benchmark_matrix(augmented_data_definition[1], n_benchmarks,
                                               self.morphing_matrix, self.morphing_components)
                )
                augmented_data_theta_matrices_den.append(
                    get_theta_benchmark_matrix(augmented_data_definition[2], n_benchmarks,
                                               self.morphing_matrix, self.morphing_components)
                )

            elif augmented_data_types[-1] == 'score':
                augmented_data_theta_matrices_num.append(
                    get_dtheta_benchmark_matrix(augmented_data_definition[1], n_benchmarks,
                                                self.morphing_matrix, self.morphing_components)
                )
                augmented_data_theta_matrices_den.append(
                    get_theta_benchmark_matrix(augmented_data_definition[1], n_benchmarks,
                                               self.morphing_matrix, self.morphing_components)
                )

        # Loop over thetas
        for theta, n_samples in zip(thetas, n_samples_per_theta):

            # Prepare output
            samples_done = np.zeros(n_samples, dtype=np.bool)
            samples_x = np.zeros((n_samples, n_observables))
            samples_augmented_data = np.zeros((n_samples, n_augmented_data))

            # Find theta_matrix
            theta_matrix = get_theta_benchmark_matrix(theta, n_benchmarks, self.morphing_matrix,
                                                      self.morphing_components)

            # Total xsec for this theta
            xsec_theta = theta_matrix.dot(xsecs_benchmarks)

            # Draw random numbers in [0, 1]
            u = np.random.rand(n_samples)

            cumulative_p = [0.]

            for x_batch, weights_benchmarks_batch in madminer_event_loader(self.madminer_filename,
                                                                           start=start_event,
                                                                           end=end_event):
                # Evaluate cumulative p(x | theta)
                weights_theta = theta_matrix.dot(weights_benchmarks_batch)  # Shape (n_events_in_batch,)
                p_theta = weights_theta / xsec_theta
                cumulative_p = cumulative_p[-1] + np.cumsum(p_theta)

                # Check what we've found
                indices = np.searchsorted(cumulative_p, u, side='left')
                found_now = ((not samples_done) & (indices < len(cumulative_p)))

                # Save x
                samples_x[found_now] = x_batch[indices[found_now]]

                # Extract augmented data
                samples_augmented_data[found_now] = extract_augmented_data(
                    augmented_data_types,
                    augmented_data_theta_matrices_num,
                    augmented_data_theta_matrices_den,
                    weights_benchmarks_batch,
                    xsecs_benchmarks
                )
