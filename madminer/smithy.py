from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import collections

from madminer.tools.h5_interface import load_madminer_settings, madminer_event_loader
from madminer.tools.analysis import get_theta_value, get_theta_benchmark_matrix, get_dtheta_benchmark_matrix
from madminer.tools.analysis import extract_augmented_data, parse_theta
from madminer.tools.morphing import Morpher


def constant_benchmark_theta(benchmark_name):
    return 'benchmark', benchmark_name


def multiple_benchmark_thetas(benchmark_names):
    return 'benchmarks', benchmark_names


def constant_morphing_theta(theta):
    return 'theta', theta


def multiple_morphing_thetas(thetas):
    return 'thetas', thetas


def random_morphing_thetas(n_thetas, priors):
    return 'random', (n_thetas, priors)


class Smithy:

    def __init__(self, filename, disable_morphing=False):
        self.madminer_filename = filename

        # Load data
        (self.parameters, self.benchmarks, self.morphing_components, self.morphing_matrix,
         self.observables, self.n_samples) = load_madminer_settings(filename)

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

        # Thetas
        theta_types, theta_values, n_samples_per_theta = parse_theta(theta, n_samples)

        # Augmented data (gold)
        augmented_data_definitions = []

        # Train / test split
        if test_split is None or test_split <= 0. or test_split >= 1.:
            last_train_index = None
        else:
            last_train_index = int(round((1. - test_split) * self.n_samples, 0))

            if last_train_index < 0 or last_train_index > self.n_samples:
                raise ValueError("Irregular train / test split: sample {} / {}", last_train_index, self.n_samples)

        # Start
        x, _, theta, _ = self.extract_sample(
            theta_sampling_types=theta_types,
            theta_sampling_values=theta_values,
            n_samples_per_theta=n_samples_per_theta,
            augmented_data_definitions=augmented_data_definitions,
            start_event=0,
            end_event=last_train_index
        )

        # Save data
        np.save(folder + '/theta_' + filename + '.npy', theta)
        np.save(folder + '/x_' + filename + '.npy', x)

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

        # Thetas
        theta_types, theta_values, n_samples_per_theta = parse_theta(theta, n_samples)

        # Augmented data (gold)
        augmented_data_definitions = ['score', ('sampling', None)]

        # Train / test split
        if test_split is None or test_split <= 0. or test_split >= 1.:
            last_train_index = None
        else:
            last_train_index = int(round((1. - test_split) * self.n_samples, 0))

            if last_train_index < 0 or last_train_index > self.n_samples:
                raise ValueError("Irregular train / test split: sample {} / {}", last_train_index, self.n_samples)

        # Start
        x, (r_xz,), theta, _ = self.extract_sample(
            theta_sampling_types=theta_types,
            theta_sampling_values=theta_values,
            n_samples_per_theta=n_samples_per_theta,
            augmented_data_definitions=augmented_data_definitions,
            start_event=0,
            end_event=last_train_index
        )

        # Save data
        np.save(folder + '/theta_' + filename + '.npy', theta)
        np.save(folder + '/x_' + filename + '.npy', x)
        np.save(folder + '/t_xz_' + filename + '.npy', r_xz)

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

        # Thetas
        theta0_types, theta0_values, n_samples_per_theta0 = parse_theta(theta0, n_samples // 2)
        theta1_types, theta1_values, n_samples_per_theta1 = parse_theta(theta1, n_samples // 2)

        # Augmented data (gold)
        augmented_data_definitions0 = [('ratio', ('sampling', None), ('auxiliary', None)),
                                       ('score', ('sampling', None))]
        augmented_data_definitions1 = [('ratio', ('auxiliary', None), ('sampling', None)),
                                       ('score', ('auxiliary', None))]

        # Train / test split
        if test_split is None or test_split <= 0. or test_split >= 1.:
            last_train_index = None
        else:
            last_train_index = int(round((1. - test_split) * self.n_samples, 0))

            if last_train_index < 0 or last_train_index > self.n_samples:
                raise ValueError("Irregular train / test split: sample {} / {}", last_train_index, self.n_samples)

        # Start for theta0
        x0, (r_xz0, t_xz0), theta0_0, theta1_0 = self.extract_sample(
            theta_sampling_types=theta0_types,
            theta_sampling_values=theta0_values,
            theta_auxiliary_types=theta1_types,
            theta_auxiliary_values=theta1_values,
            n_samples_per_theta=n_samples_per_theta0,
            augmented_data_definitions=augmented_data_definitions0,
            start_event=0,
            end_event=last_train_index
        )

        # Start for theta1
        x1, (r_xz1, t_xz1), theta1_1, theta0_1 = self.extract_sample(
            theta_sampling_types=theta1_types,
            theta_sampling_values=theta1_values,
            theta_auxiliary_types=theta0_types,
            theta_auxiliary_values=theta0_values,
            n_samples_per_theta=n_samples_per_theta1,
            augmented_data_definitions=augmented_data_definitions1,
            start_event=0,
            end_event=last_train_index
        )

        # Combine
        x = np.vstack([x0, x1])
        r_xz = np.vstack([r_xz0, r_xz1])
        t_xz = np.vstack([t_xz0, t_xz1])
        theta0 = np.vstack([theta0_0, theta0_1])
        theta1 = np.vstack([theta1_0, theta1_1])
        y = np.zeros(x.shape[0])
        y[x0.shape[0]:] = 1.

        # Shuffle
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation]
        r_xz = r_xz[permutation]
        t_xz = t_xz[permutation]
        theta0 = theta0[permutation]
        theta1 = theta1[permutation]
        y = y[permutation]

        # Save data
        np.save(folder + '/theta0_' + filename + '.npy', theta0)
        np.save(folder + '/theta1_' + filename + '.npy', theta1)
        np.save(folder + '/x_' + filename + '.npy', x)
        np.save(folder + '/y_' + filename + '.npy', y)
        np.save(folder + '/r_xz_' + filename + '.npy', r_xz)
        np.save(folder + '/t_xz_' + filename + '.npy', t_xz)

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

        # Thetas
        theta_types, theta_values, n_samples_per_theta = parse_theta(theta, n_samples)

        # Augmented data (gold)
        augmented_data_definitions = []

        # Train / test split
        if test_split is None or test_split <= 0. or test_split >= 1.:
            first_test_index = 0
        else:
            first_test_index = int(round((1. - test_split) * self.n_samples, 0)) + 1

            if first_test_index < 0 or first_test_index > self.n_samples:
                raise ValueError("Irregular in train / test split: sample {} / {}", first_test_index, self.n_samples)

        # Extract information
        x, _, theta, _ = self.extract_sample(
            theta_sampling_types=theta_types,
            theta_sampling_values=theta_values,
            n_samples_per_theta=n_samples_per_theta,
            augmented_data_definitions=augmented_data_definitions,
            start_event=first_test_index,
            end_event=None
        )

        # Save data
        np.save(folder + '/theta_' + filename + '.npy', theta)
        np.save(folder + '/x_' + filename + '.npy', x)

    def extract_sample(self,
                       theta_sampling_types,
                       theta_sampling_values,
                       n_samples_per_theta,
                       theta_auxiliary_types=None,
                       theta_auxiliary_values=None,
                       augmented_data_definitions=None,
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
        :param theta_auxiliary_values:
        :param theta_auxiliary_types:
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
        augmented_data_sizes = []

        if augmented_data_definitions is None:
            augmented_data_definitions = []

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
                augmented_data_sizes.append(1)

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
                augmented_data_sizes.append(len(self.parameters))

        # Samples per theta
        if not isinstance(n_samples_per_theta, collections.Iterable):
            n_samples_per_theta = [n_samples_per_theta] * len(theta_sampling_types)
        elif len(n_samples_per_theta) == 1:
            n_samples_per_theta = [n_samples_per_theta[0]] * len(theta_sampling_types)

        # Auxiliary thetas
        if theta_auxiliary_types is None or theta_auxiliary_values is None:
            theta_auxiliary_types = [None for _ in theta_sampling_types]
            theta_auxiliary_values = [None for _ in theta_sampling_values]

        # Prepare output
        all_x = []
        all_augmented_data = ([] for _ in range(n_augmented_data))
        all_theta_sampling = []
        all_theta_auxiliary = []

        # Loop over thetas
        for (theta_sampling_type, theta_sampling_value, n_samples, theta_auxiliary_type,
             theta_auxiliary_value) in zip(theta_sampling_types, theta_sampling_values, n_samples_per_theta,
                                           theta_auxiliary_types, theta_auxiliary_values):

            # Prepare output
            samples_done = np.zeros(n_samples, dtype=np.bool)
            samples_x = np.zeros((n_samples, n_observables))
            samples_augmented_data = (np.zeros((n_samples, augmented_data_sizes[i])) for i in range(n_augmented_data))

            # Sampling theta
            theta_sampling = get_theta_value(theta_sampling_type, theta_sampling_value, self.benchmarks)
            theta_sampling = np.broadcast_to(theta_sampling, (n_samples, theta_sampling.size))
            theta_sampling_matrix = get_theta_benchmark_matrix(theta_sampling_type, theta_sampling_value, n_benchmarks,
                                                               self.morpher)

            # Total xsec for this theta
            xsec_sampling_theta = theta_sampling_matrix.dot(xsecs_benchmarks)

            # Auxiliary theta
            if theta_auxiliary_type is not None:
                theta_auxiliary = get_theta_value(theta_auxiliary_type, theta_auxiliary_value, self.benchmarks)
                theta_auxiliary = np.broadcast_to(theta_auxiliary, (n_samples, theta_auxiliary.size))
                theta_auxiliary_matrix = get_theta_benchmark_matrix(theta_auxiliary_type, theta_auxiliary_value,
                                                                    n_benchmarks, self.morpher)
            else:
                theta_auxiliary = None
                theta_auxiliary_matrix = None

            # Draw random numbers in [0, 1]
            u = np.random.rand(n_samples)

            cumulative_p = [0.]

            for x_batch, weights_benchmarks_batch in madminer_event_loader(self.madminer_filename,
                                                                           start=start_event,
                                                                           end=end_event):
                # Evaluate cumulative p(x | theta)
                weights_theta = theta_sampling_matrix.dot(weights_benchmarks_batch.T)  # Shape (n_events_in_batch,)
                p_theta = weights_theta / xsec_sampling_theta
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
                    xsecs_benchmarks,
                    theta_sampling_matrix,
                    theta_auxiliary_matrix
                )

                for i, this_relevant_augmented_data in enumerate(relevant_augmented_data):
                    samples_augmented_data[i][found_now] = this_relevant_augmented_data

                samples_done[found_now] = True

                if np.all(samples_done):
                    break

            # Check that we got 'em all
            if not np.all(samples_done):
                raise ValueError('{} / {} samples not found, u = {}', np.sum(np.invert(samples_done)),
                                 samples_done.size, u[np.invert(samples_done)])

            all_x.append(samples_x)
            all_theta_sampling.append(theta_sampling)
            all_theta_auxiliary.append(theta_auxiliary)
            for i, this_samples_augmented_data in enumerate(samples_augmented_data):
                all_augmented_data[i].append(this_samples_augmented_data)

        all_x = np.vstack(all_x)
        all_theta_sampling = np.vstack(all_theta_sampling)
        all_theta_auxiliary = np.vstack(all_theta_auxiliary)
        for i in range(n_augmented_data):
            all_augmented_data[i] = np.vstack(all_augmented_data[i])

        return all_x, all_augmented_data, all_theta_sampling, all_theta_auxiliary
