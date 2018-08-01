from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import collections
import six

from madminer.tools.h5_interface import load_madminer_settings, madminer_event_loader, save_events_to_madminer_file
from madminer.tools.analysis import get_theta_value, get_theta_benchmark_matrix, get_dtheta_benchmark_matrix
from madminer.tools.analysis import extract_augmented_data, parse_theta
from madminer.tools.morphing import Morpher
from madminer.tools.utils import general_init, format_benchmark, create_missing_folders, shuffle


def combine_and_shuffle(input_filenames,
                        output_filename,
                        overwrite_existing_file=True,
                        debug=False):
    """
    Combines multiple HDF5 files into one, and shuffles the order of the events. It is recommended to run this tool
    before the Refinery.

    :param input_filenames: list of filenames of the input HDF5 files
    :param output_filename: filename for the output HDF5 file
    :param overwrite_existing_file:
    :param debug:
    """

    general_init(debug=debug)

    if len(input_filenames) > 1:
        logging.warning('Careful: this tool assumes that all samples are generated with the same setup, including'
                        ' identical benchmarks (and thus morphing setup). If it is used with samples with different'
                        ' settings, there will be wrong results! There are no explicit cross checks in place yet.')

    # Copy first file to output_filename
    logging.info('Copying setup from %s to %s', input_filenames[0], output_filename)

    # TODO: More memory efficient strategy

    # Load events
    all_observations = None
    all_weights = None

    for i, filename in enumerate(input_filenames):
        logging.info('Loading samples from file %s / %s at %s', i + 1, len(input_filenames), filename)

        for observations, weights in madminer_event_loader(filename):
            if all_observations is None:
                all_observations = observations
                all_weights = weights
            else:
                all_observations = np.vstack((all_observations, observations))
                all_weights = np.vstack((all_weights, weights))

    # Shuffle
    all_observations, all_weights = shuffle(all_observations, all_weights)

    # Save result
    save_events_to_madminer_file(
        filename=output_filename,
        observations=all_observations,
        weights=all_weights,
        copy_setup_from=input_filenames[0],
        overwrite_existing_samples=overwrite_existing_file
    )


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


class Refinery:

    def __init__(self, filename, disable_morphing=False, debug=False):

        general_init(debug=debug)

        self.madminer_filename = filename

        logging.info('Loading data from %s', filename)

        # Load data
        (self.parameters, self.benchmarks, self.morphing_components, self.morphing_matrix,
         self.observables, self.n_samples) = load_madminer_settings(filename)

        logging.info('Found %s parameters:', len(self.parameters))
        for key, values in six.iteritems(self.parameters):
            logging.info('   %s (LHA: %s %s, maximal power in squared ME: %s, range: %s)',
                         key, values[0], values[1], values[2], values[3])

        logging.info('Found %s benchmarks:', len(self.benchmarks))
        for key, values in six.iteritems(self.benchmarks):
            logging.info('   %s: %s',
                         key, format_benchmark(values))

        logging.info('Found %s observables: %s', len(self.observables), ', '.join(self.observables))
        logging.info('Found %s events', self.n_samples)

        # Morphing
        self.morpher = None
        if self.morphing_matrix is not None and self.morphing_components is not None and not disable_morphing:
            self.morpher = Morpher(self.parameters)
            self.morpher.set_components(self.morphing_components)
            self.morpher.set_basis(self.benchmarks, morphing_matrix=self.morphing_matrix)

            logging.info('Found morphing setup with %s components', len(self.morphing_components))

        else:
            logging.info('Did not find morphing setup.')

    def extract_samples_train_plain(self,
                                    theta,
                                    n_samples,
                                    folder,
                                    filename,
                                    test_split=0.3):
        """
        Extracts training samples x ~ p(x|theta) for methods such as histograms or ABC.

        :param theta: tuple (type, value) that defines the parameter point or prior over parameter points for the
                      sampling. Use the helper functions constant_benchmark_theta(), multiple_benchmark_thetas(),
                      constant_morphing_theta(), multiple_morphing_thetas(), or random_morphing_thetas().
        :param n_samples: Total number of samples to be drawn.
        :param folder: Folder for the resulting samples.
        :param filename: Label for the filenames. The actual filenames will add a prefix such as 'x_', and the extension
                         '.npy'.
        :param test_split: Fraction of events reserved for the test sample (will not be used for any training samples).
        """

        logging.info('Extracting plain training sample. Sampling according to %s', theta)

        create_missing_folders([folder])

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

        return x, theta

    def extract_samples_train_local(self,
                                    theta,
                                    n_samples,
                                    folder,
                                    filename,
                                    test_split=0.3):
        """
        Extracts training samples x ~ p(x|theta) as well as the joint score t(x, z|theta) for SALLY and SALLINO.

        :param theta: tuple (type, value) that defines the parameter point or prior over parameter points for the
                      sampling. This is also where the score is evaluated. Use the helper functions, in particular
                      constant_benchmark_theta() and constant_morphing_theta().
        :param n_samples: Total number of samples to be drawn.
        :param folder: Folder for the resulting samples.
        :param filename: Label for the filenames. The actual filenames will add a prefix such as 'x_', and the extension
                         '.npy'.
        :param test_split: Fraction of events reserved for the test sample (will not be used for any training samples).
        """

        logging.info('Extracting training sample for local score regression. Sampling and score evaluation according to'
                     ' %s', theta)

        create_missing_folders([folder])

        if self.morpher is None:
            raise RuntimeError('No morphing setup loaded. Cannot calculate score.')

        # Thetas
        theta_types, theta_values, n_samples_per_theta = parse_theta(theta, n_samples)

        # Augmented data (gold)
        augmented_data_definitions = [('score', 'sampling', None)]

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

        return x, theta, r_xz

    def extract_samples_train_ratio(self,
                                    theta0,
                                    theta1,
                                    n_samples,
                                    folder,
                                    filename,
                                    test_split=0.3):
        """
        Extracts training samples x ~ p(x|theta0) and x ~ p(x|theta1) together with the class label y, the joint
        likelihood ratio r(x,z|theta0, theta1), and the joint score t(x,z|theta0) for methods such as CARL, ROLR,
        CASCAL, and RASCAL.

        :param theta0: tuple (type, value) that defines the numerator parameter point or prior over parameter points.
                       Use the helper functions constant_benchmark_theta(), multiple_benchmark_thetas(),
                       constant_morphing_theta(), multiple_morphing_thetas(), or random_morphing_thetas().
        :param theta1: tuple (type, value) that defines the numerator parameter point or prior over parameter points.
                       Use the helper functions constant_benchmark_theta(), multiple_benchmark_thetas(),
                       constant_morphing_theta(), multiple_morphing_thetas(), or random_morphing_thetas().
        :param n_samples: Total number of samples to be drawn.
        :param folder: Folder for the resulting samples.
        :param filename: Label for the filenames. The actual filenames will add a prefix such as 'x_', and the extension
                         '.npy'.
        :param test_split: Fraction of events reserved for the test sample (will not be used for any training samples).
        """

        logging.info('Extracting training sample for ratio-based methods. Numerator hypothesis: %s, denominator '
                     'hypothesis: %s', theta0, theta1)

        if self.morpher is None:
            raise RuntimeError('No morphing setup loaded. Cannot calculate score.')

        create_missing_folders([folder])

        # Augmented data (gold)
        augmented_data_definitions0 = [('ratio', 'sampling', None, 'auxiliary', None),
                                       ('score', 'sampling', None)]
        augmented_data_definitions1 = [('ratio', 'auxiliary', None, 'sampling', None),
                                       ('score', 'auxiliary', None)]

        # Train / test split
        if test_split is None or test_split <= 0. or test_split >= 1.:
            last_train_index = None
        else:
            last_train_index = int(round((1. - test_split) * self.n_samples, 0))

            if last_train_index < 0 or last_train_index > self.n_samples:
                raise ValueError("Irregular train / test split: sample {} / {}", last_train_index, self.n_samples)

        # Thetas for theta0 sampling
        theta0_types, theta0_values, n_samples_per_theta0 = parse_theta(theta0, n_samples // 2)
        theta1_types, theta1_values, n_samples_per_theta1 = parse_theta(theta1, n_samples // 2)

        n_samples_per_theta = min(n_samples_per_theta0, n_samples_per_theta1)

        # Start for theta0
        x0, (r_xz0, t_xz0), theta0_0, theta1_0 = self.extract_sample(
            theta_sampling_types=theta0_types,
            theta_sampling_values=theta0_values,
            theta_auxiliary_types=theta1_types,
            theta_auxiliary_values=theta1_values,
            n_samples_per_theta=n_samples_per_theta,
            augmented_data_definitions=augmented_data_definitions0,
            start_event=0,
            end_event=last_train_index
        )

        # Thetas for theta1 sampling (could be different if num or denom are random)
        theta0_types, theta0_values, n_samples_per_theta0 = parse_theta(theta0, n_samples // 2)
        theta1_types, theta1_values, n_samples_per_theta1 = parse_theta(theta1, n_samples // 2)

        n_samples_per_theta = min(n_samples_per_theta0, n_samples_per_theta1)

        # Start for theta1
        x1, (r_xz1, t_xz1), theta1_1, theta0_1 = self.extract_sample(
            theta_sampling_types=theta1_types,
            theta_sampling_values=theta1_values,
            theta_auxiliary_types=theta0_types,
            theta_auxiliary_values=theta0_values,
            n_samples_per_theta=n_samples_per_theta,
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

        # y shape
        y = y.reshape((-1, 1))

        # Save data
        np.save(folder + '/theta0_' + filename + '.npy', theta0)
        np.save(folder + '/theta1_' + filename + '.npy', theta1)
        np.save(folder + '/x_' + filename + '.npy', x)
        np.save(folder + '/y_' + filename + '.npy', y)
        np.save(folder + '/r_xz_' + filename + '.npy', r_xz)
        np.save(folder + '/t_xz_' + filename + '.npy', t_xz)

        return x, theta0, theta1, y, r_xz, t_xz

    def extract_samples_train_ratio_double(self,
                                           theta0,
                                           theta1,
                                           n_samples,
                                           folder,
                                           filename,
                                           additional_theta_eval=None,
                                           test_split=0.3):
        """
        Extracts training samples x ~ p(x|theta0) and x ~ p(x|theta1) together with the class label y, the joint
        likelihood ratio r(x,z|theta0, theta1), and the joint score t(x,z|theta0) for methods such as CARL, ROLR,
        CASCAL, and RASCAL.

        :param theta0: tuple (type, value) that defines the numerator parameter point or prior over parameter points.
                       Use the helper functions constant_benchmark_theta(), multiple_benchmark_thetas(),
                       constant_morphing_theta(), multiple_morphing_thetas(), or random_morphing_thetas().
        :param theta1: tuple (type, value) that defines the numerator parameter point or prior over parameter points.
                       Use the helper functions constant_benchmark_theta(), multiple_benchmark_thetas(),
                       constant_morphing_theta(), multiple_morphing_thetas(), or random_morphing_thetas().
        :param n_samples: Total number of samples to be drawn.
        :param folder: Folder for the resulting samples.
        :param filename: Label for the filenames. The actual filenames will add a prefix such as 'x_', and the extension
                         '.npy'.
        :param test_split: Fraction of events reserved for the test sample (will not be used for any training samples).
        """

        raise NotImplementedError

        logging.info('Extracting training sample for ratio-based methods. Numerator hypothesis: %s, denominator '
                     'hypothesis: %s', theta0, theta1)

        if self.morpher is None:
            raise RuntimeError('No morphing setup loaded. Cannot calculate score.')

        create_missing_folders([folder])

        # Augmented data (gold)
        augmented_data_definitions0 = [('ratio', 'sampling', None, 'auxiliary', None),
                                       ('score', 'sampling', None),
                                       ('score', 'auxiliary', None)]
        augmented_data_definitions1 = [('ratio', 'auxiliary', None, 'sampling', None),
                                       ('score', 'auxiliary', None),
                                       ('score', 'sampling', None)]

        # Additional evaluation thetas
        if additional_theta_eval is not None:
            theta_eval_types, theta_eval_values, _ = parse_theta(theta_eval, 1)

            for theta_eval_type, theta_eval_value in zip(theta_eval_types, theta_eval_values):
                augmented_data_definitions0.append(
                    ('ratio', 'sampling', None, 'morphing', (theta_eval_type, theta_eval_value))
                )
                augmented_data_definitions0.append(
                    ('score', 'morphing', (theta_eval_type, theta_eval_value))
                )
                augmented_data_definitions1.append(
                    ('ratio', 'morphing', (theta_eval_type, theta_eval_value), 'sampling', None)
                )
                augmented_data_definitions1.append(
                    ('score', 'morphing', (theta_eval_type, theta_eval_value))
                )

        # Train / test split
        if test_split is None or test_split <= 0. or test_split >= 1.:
            last_train_index = None
        else:
            last_train_index = int(round((1. - test_split) * self.n_samples, 0))

            if last_train_index < 0 or last_train_index > self.n_samples:
                raise ValueError("Irregular train / test split: sample {} / {}", last_train_index, self.n_samples)

        # Thetas for theta0 sampling
        theta0_types, theta0_values, n_samples_per_theta0 = parse_theta(theta0, n_samples // 2)
        theta1_types, theta1_values, n_samples_per_theta1 = parse_theta(theta1, n_samples // 2)

        n_samples_per_theta = min(n_samples_per_theta0, n_samples_per_theta1)

        # TODO: Handle additional augmented data

        # Start for theta0
        x0, (r_xz0, t_xz0), theta0_0, theta1_0 = self.extract_sample(
            theta_sampling_types=theta0_types,
            theta_sampling_values=theta0_values,
            theta_auxiliary_types=theta1_types,
            theta_auxiliary_values=theta1_values,
            n_samples_per_theta=n_samples_per_theta,
            augmented_data_definitions=augmented_data_definitions0,
            start_event=0,
            end_event=last_train_index
        )

        # Thetas for theta1 sampling (could be different if num or denom are random)
        theta0_types, theta0_values, n_samples_per_theta0 = parse_theta(theta0, n_samples // 2)
        theta1_types, theta1_values, n_samples_per_theta1 = parse_theta(theta1, n_samples // 2)

        n_samples_per_theta = min(n_samples_per_theta0, n_samples_per_theta1)

        # Start for theta1
        x1, (r_xz1, t_xz1), theta1_1, theta0_1 = self.extract_sample(
            theta_sampling_types=theta1_types,
            theta_sampling_values=theta1_values,
            theta_auxiliary_types=theta0_types,
            theta_auxiliary_values=theta0_values,
            n_samples_per_theta=n_samples_per_theta,
            augmented_data_definitions=augmented_data_definitions1,
            start_event=0,
            end_event=last_train_index
        )

        # TODO: Copy events with additional theta_not_sampling / r_xz / t_xz

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

        # y shape
        y = y.reshape((-1, 1))

        # Save data
        np.save(folder + '/theta0_' + filename + '.npy', theta0)
        np.save(folder + '/theta1_' + filename + '.npy', theta1)
        np.save(folder + '/x_' + filename + '.npy', x)
        np.save(folder + '/y_' + filename + '.npy', y)
        np.save(folder + '/r_xz_' + filename + '.npy', r_xz)
        np.save(folder + '/t_xz_' + filename + '.npy', t_xz)

        return x, theta0, theta1, y, r_xz, t_xz

    def extract_samples_test(self,
                             theta,
                             n_samples,
                             folder,
                             filename,
                             test_split=0.3):
        """
        Extracts evaluation samples x ~ p(x|theta).

        :param theta: tuple (type, value) that defines the parameter point or prior over parameter points used for the
                      sampling. Use the helper functions constant_benchmark_theta(), multiple_benchmark_thetas(),
                      constant_morphing_theta(), multiple_morphing_thetas(), or random_morphing_thetas().
        :param n_samples: Total number of samples to be drawn.
        :param folder: Folder for the resulting samples.
        :param filename: Label for the filenames. The actual filenames will add a prefix such as 'x_', and the extension
                         '.npy'.
        :param test_split: Fraction of events reserved for this evaluation sample (will not be used for any training
                           samples).
        """

        logging.info('Extracting evaluation sample. Sampling according to %s', theta)

        create_missing_folders([folder])

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

        return x, theta

    def extract_cross_sections(self,
                               theta):

        """
        Calculates the total cross sections for all specified thetas.

        :param theta: tuple (type, value) that defines the parameter point or prior over parameter points used for the
                      sampling. Use the helper functions constant_benchmark_theta(), multiple_benchmark_thetas(),
                      constant_morphing_theta(), multiple_morphing_thetas(), or random_morphing_thetas().
        :return: thetas, xsecs, xsec_uncertainties. xsecs and xsec_uncertainties are in pb.
        """

        # Total xsecs for benchmarks
        xsecs_benchmarks = None
        squared_weight_sum_benchmarks = None

        for obs, weights in madminer_event_loader(self.madminer_filename):
            if xsecs_benchmarks is None:
                xsecs_benchmarks = np.sum(weights, axis=0)
                squared_weight_sum_benchmarks = np.sum(weights * weights, axis=0)
            else:
                xsecs_benchmarks += np.sum(weights, axis=0)
                squared_weight_sum_benchmarks += np.sum(weights * weights, axis=0)

        # Parse thetas for evaluation
        theta_types, theta_values, _ = parse_theta(theta, 1)

        # Loop over thetas
        all_thetas = []
        all_xsecs = []
        all_xsec_uncertainties = []

        for (theta_type, theta_value) in zip(theta_types, theta_values):

            if self.morpher is None and theta_type == 'morphing':
                raise RuntimeError('Theta defined through morphing, but no morphing setup has been loaded.')

            theta = get_theta_value(theta_type, theta_value, self.benchmarks)
            theta_matrix = get_theta_benchmark_matrix(
                theta_type,
                theta_value,
                self.benchmarks,
                self.morpher
            )

            # Total xsec for this theta
            xsec_theta = theta_matrix.dot(xsecs_benchmarks)
            rms_xsec_theta = ((theta_matrix * theta_matrix).dot(squared_weight_sum_benchmarks)) ** 0.5

            all_thetas.append(theta)
            all_xsecs.append(xsec_theta)
            all_xsec_uncertainties.append(rms_xsec_theta)

            logging.info('theta %s: xsec = (%s +/- %s) pb', theta, xsec_theta, rms_xsec_theta)

        # Return
        all_thetas = np.array(all_thetas)
        all_xsecs = np.array(all_xsecs)
        all_xsec_uncertainties = np.array(all_xsec_uncertainties)

        return all_thetas, all_xsecs, all_xsec_uncertainties

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

        :param theta_sampling_types: list of str, each entry can be 'benchmark' or 'morphing'
        :param theta_sampling_values: list, each entry is int and labels the benchmark index (if the corresponding
                                      theta_sampling_types entry is 'benchmark') or a numpy array with the theta values
                                      (of the corresponding theta_sampling_types entry is 'morphing')
        :param n_samples_per_theta: Number of samples to be drawn per entry in theta_sampling_types.
        :param augmented_data_definitions: list of tuples. Each tuple can either be ('ratio', num_type, num_value,
                                           den_type, den_value) or ('score', theta_type, theta_value). The theta types
                                           can either be 'benchmark', 'morphing', 'sampling', or 'auxiliary'. The
                                           corresponding theta values are then either the benchmark name, the theta
                                           value, None, or None.
        :param theta_auxiliary_values: list of str, each entry can be 'benchmark' or 'morphing'
        :param theta_auxiliary_types: list, each entry is int and labels the benchmark index (if the corresponding
                                      theta_sampling_types entry is 'benchmark') or a numpy array with the theta values
                                      (of the corresponding theta_sampling_types entry is 'morphing')
        :param start_event: Index of first event to consider.
        :param end_event: Index of last event to consider.
        :return: tuple (x, augmented_data_list, theta_sampling, theta_auxiliary). x, theta_sampling, theta_auxiliary,
                 and all elements of the list augmented_data_list are ndarrays with the number of samples as first
                 dimension.
        """

        # Calculate total xsecs
        xsecs_benchmarks = None
        squared_weight_sum_benchmarks = None
        n_observables = 0

        for obs, weights in madminer_event_loader(self.madminer_filename, start=start_event, end=end_event):
            if xsecs_benchmarks is None:
                xsecs_benchmarks = np.sum(weights, axis=0)
                squared_weight_sum_benchmarks = np.sum(weights * weights, axis=0)
            else:
                xsecs_benchmarks += np.sum(weights, axis=0)
                squared_weight_sum_benchmarks += np.sum(weights * weights, axis=0)

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

        logging.info('Augmented data requested:')

        for augmented_data_definition in augmented_data_definitions:
            augmented_data_types.append(augmented_data_definition[0])

            if augmented_data_types[-1] == 'ratio':
                augmented_data_theta_matrices_num.append(
                    get_theta_benchmark_matrix(
                        augmented_data_definition[1],
                        augmented_data_definition[2],
                        self.benchmarks,
                        self.morpher
                    )
                )
                augmented_data_theta_matrices_den.append(
                    get_theta_benchmark_matrix(
                        augmented_data_definition[3],
                        augmented_data_definition[4],
                        self.benchmarks,
                        self.morpher
                    )
                )
                augmented_data_sizes.append(1)

                logging.debug('  Joint ratio, num %s %s, den %s %s',
                              augmented_data_definition[1],
                              augmented_data_definition[2],
                              augmented_data_definition[3],
                              augmented_data_definition[4])

            elif augmented_data_types[-1] == 'score':
                if self.morpher is None:
                    raise RuntimeError('No morphing setup loaded. Cannot calculate score.')

                augmented_data_theta_matrices_num.append(
                    get_dtheta_benchmark_matrix(
                        augmented_data_definition[1],
                        augmented_data_definition[2],
                        self.benchmarks,
                        self.morpher
                    )
                )
                augmented_data_theta_matrices_den.append(
                    get_theta_benchmark_matrix(
                        augmented_data_definition[1],
                        augmented_data_definition[2],
                        self.benchmarks,
                        self.morpher
                    )
                )
                augmented_data_sizes.append(len(self.parameters))

                logging.debug('  Joint score, at %s %s',
                              augmented_data_definition[1],
                              augmented_data_definition[2])

            else:
                logging.warning("Unknown augmented data type %s", augmented_data_types[-1])

        # Auxiliary thetas
        if theta_auxiliary_types is None or theta_auxiliary_values is None:
            theta_auxiliary_types = [None for _ in theta_sampling_types]
            theta_auxiliary_values = [None for _ in theta_sampling_values]

        logging.debug('Sampling and auxiliary thetas before balancing:')
        logging.debug('  sampling thetas:  %s types, %s values', len(theta_sampling_types), len(theta_sampling_values))
        logging.debug('  auxiliary thetas: %s types, %s values', len(theta_auxiliary_types),
                      len(theta_auxiliary_values))

        if len(theta_auxiliary_types) < len(theta_sampling_types):
            theta_auxiliary_types = [theta_auxiliary_types[i % len(theta_auxiliary_types)]
                                     for i in range(len(theta_sampling_types))]
            theta_auxiliary_values = [theta_auxiliary_values[i % len(theta_auxiliary_values)]
                                      for i in range(len(theta_sampling_types))]
        elif len(theta_auxiliary_types) > len(theta_sampling_types):
            theta_sampling_types = [theta_sampling_types[i % len(theta_sampling_types)]
                                    for i in range(len(theta_auxiliary_types))]
            theta_sampling_values = [theta_sampling_values[i % len(theta_sampling_values)]
                                     for i in range(len(theta_auxiliary_types))]

        # Samples per theta
        if not isinstance(n_samples_per_theta, collections.Iterable):
            n_samples_per_theta = [n_samples_per_theta] * len(theta_sampling_types)
        elif len(n_samples_per_theta) == 1:
            n_samples_per_theta = [n_samples_per_theta[0]] * len(theta_sampling_types)

        logging.debug('Sampling and auxiliary thetas after balancing:')
        logging.debug('  sampling thetas:  %s types, %s values', len(theta_sampling_types), len(theta_sampling_values))
        logging.debug('  auxiliary thetas: %s types, %s values', len(theta_auxiliary_types),
                      len(theta_auxiliary_values))

        assert (len(theta_sampling_types) == len(theta_sampling_values)
                == len(theta_auxiliary_values) == len(theta_auxiliary_types))

        # Prepare output
        all_x = []
        all_augmented_data = [[] for _ in range(n_augmented_data)]
        all_theta_sampling = []
        all_theta_auxiliary = []

        # Loop over thetas
        for (theta_sampling_type, theta_sampling_value, n_samples, theta_auxiliary_type,
             theta_auxiliary_value) in zip(theta_sampling_types, theta_sampling_values, n_samples_per_theta,
                                           theta_auxiliary_types, theta_auxiliary_values):

            if self.morpher is None and (theta_sampling_type == 'morphing' or theta_auxiliary_type == 'morphing'):
                raise RuntimeError('Theta defined through morphing, but no morphing setup has been loaded.')

            # Debug
            logging.debug(
                'Sampling {} samples from {} theta {} (compare to {} theta {})'.format(n_samples, theta_sampling_type,
                                                                                       theta_sampling_value,
                                                                                       theta_auxiliary_type,
                                                                                       theta_auxiliary_value))

            # Sampling theta
            theta_sampling = get_theta_value(theta_sampling_type, theta_sampling_value, self.benchmarks)
            theta_sampling = np.broadcast_to(theta_sampling, (n_samples, theta_sampling.size))
            theta_sampling_matrix = get_theta_benchmark_matrix(
                theta_sampling_type,
                theta_sampling_value,
                self.benchmarks,
                self.morpher
            )
            theta_sampling_gradients_matrix = get_dtheta_benchmark_matrix(
                theta_sampling_type,
                theta_sampling_value,
                self.benchmarks,
                self.morpher
            )

            # Total xsec for this theta
            xsec_sampling_theta = theta_sampling_matrix.dot(xsecs_benchmarks)
            rms_xsec_sampling_theta = ((theta_sampling_matrix * theta_sampling_matrix).dot(
                squared_weight_sum_benchmarks)) ** 0.5

            logging.debug('  xsec: (%s +/- %s) pb', xsec_sampling_theta, rms_xsec_sampling_theta)

            if rms_xsec_sampling_theta > 0.1 * xsec_sampling_theta:
                logging.warning('Warning: large statistical uncertainty on the total cross section for theta = %s: '
                                '(%s +/- %s) pb',
                                get_theta_value(theta_sampling_type, theta_sampling_value, self.benchmarks),
                                xsec_sampling_theta,
                                rms_xsec_sampling_theta)

            # Auxiliary theta
            if theta_auxiliary_type is not None:
                theta_auxiliary = get_theta_value(theta_auxiliary_type, theta_auxiliary_value, self.benchmarks)
                theta_auxiliary = np.broadcast_to(theta_auxiliary, (n_samples, theta_auxiliary.size))
                theta_auxiliary_matrix = get_theta_benchmark_matrix(
                    theta_auxiliary_type,
                    theta_auxiliary_value,
                    self.benchmarks,
                    self.morpher
                )
                theta_auxiliary_gradients_matrix = get_dtheta_benchmark_matrix(
                    theta_auxiliary_type,
                    theta_auxiliary_value,
                    self.benchmarks,
                    self.morpher
                )
            else:
                theta_auxiliary = None
                theta_auxiliary_matrix = None
                theta_auxiliary_gradients_matrix = None

            # Prepare output
            samples_done = np.zeros(n_samples, dtype=np.bool)
            samples_x = np.zeros((n_samples, n_observables))
            samples_augmented_data = [np.zeros((n_samples, augmented_data_sizes[i])) for i in range(n_augmented_data)]

            # Draw random numbers in [0, 1]

            while not np.all(samples_done):
                u = np.random.rand(n_samples)  # Shape: (n_samples,)

                cumulative_p = np.array([0.])

                for x_batch, weights_benchmarks_batch in madminer_event_loader(self.madminer_filename,
                                                                               start=start_event,
                                                                               end=end_event):
                    # Evaluate cumulative p(x | theta)
                    weights_theta = theta_sampling_matrix.dot(weights_benchmarks_batch.T)  # Shape (n_batch_size,)
                    p_theta = weights_theta / xsec_sampling_theta  # Shape: (n_batch_size,)

                    n_negative_weights = np.sum(p_theta < 0.)
                    if n_negative_weights > 0:
                        logging.warning('%s negative weights (%s)',
                                        n_negative_weights, n_negative_weights / p_theta.size)

                    p_theta[p_theta < 0.] = 0.

                    cumulative_p = cumulative_p.flatten()[-1] + np.cumsum(p_theta)  # Shape: (n_batch_size,)

                    # Check what we've found
                    indices = np.searchsorted(cumulative_p, u,
                                              side='left').flatten()  # Shape: (n_samples,), values: [0, ..., n_batch_size]

                    found_now = (np.invert(samples_done) & (indices < len(cumulative_p)))  # Shape: (n_samples,)

                    # # Debug
                    # logging.debug('New batch:')
                    # logging.debug("  weights: %s\n%s", weights_theta.shape, weights_theta)
                    # logging.debug("  p: %s\n%s", p_theta.shape, p_theta)
                    # logging.debug("  Cumulative p: %s\n%s", cumulative_p.shape, cumulative_p)
                    # logging.debug("  Samples done: %s\n%s", samples_done.shape, samples_done)
                    # logging.debug("  Found now: %s\n%s", found_now.shape, found_now)
                    # logging.debug("  Indices: %s, %s ... %s\n%s", indices.shape, np.min(indices), np.max(indices), indices)
                    # logging.debug("  Indices [found now]: %s, %s ... %s\n%s", indices[found_now].shape, np.min(indices[found_now]), np.max(indices[found_now]), indices[found_now])
                    # logging.debug("  Samples x: %s\n%s", samples_x.shape, samples_x)
                    # logging.debug("  x batch: %s\n%s", x_batch.shape, x_batch)

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
                        theta_sampling_gradients_matrix,
                        theta_auxiliary_matrix,
                        theta_auxiliary_gradients_matrix
                    )

                    for i, this_relevant_augmented_data in enumerate(relevant_augmented_data):
                        samples_augmented_data[i][found_now] = this_relevant_augmented_data

                    samples_done[found_now] = True

                    # if np.all(samples_done):
                    #    break

                # Check cumulative probabilities at end. Should be one!
                logging.debug('  Cumulative probability (should be close to 1): %s', cumulative_p[-1])

                # Check that we got 'em all
                if not np.all(samples_done):
                    logging.debug(
                        'After full pass through event files, {} / {} samples not found, u = {}'.format(
                            np.sum(np.invert(samples_done)),
                            samples_done.size,
                            u[np.invert(samples_done)]
                        ))

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

        logging.debug('Combined x shape: %s', all_x.shape)

        return all_x, all_augmented_data, all_theta_sampling, all_theta_auxiliary

    def extract_raw_data(self, theta=None):

        """

        :param theta: if not None, uses morphing to calculate the weights for this value of theta. If None, returns
                      the weights in fb for all benchmark points, as in the file.
        :return: x, weights
        """

        x, weights_benchmarks = next(madminer_event_loader(self.madminer_filename, batch_size=None))

        if theta is not None:
            theta_matrix = get_theta_benchmark_matrix(
                'morphing',
                theta,
                self.benchmarks,
                self.morpher
            )

            weights_theta = theta_matrix.dot(weights_benchmarks.T)

            return x, weights_theta

        return x, weights_benchmarks
