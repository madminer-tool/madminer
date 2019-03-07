from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import logging

logger = logging.getLogger(__name__)


class Histo:
    def __init__(self, n_bins_theta, n_bins_x, separate_1d_histos=False):
        self.n_bins_theta = n_bins_theta
        self.n_bins_x = n_bins_x
        self.separate_1d_x_histos = separate_1d_histos

        logger.info('Initialized histogram with the following settings:')
        logger.info('  Bins per parameter:  %s', self.n_bins_theta)
        logger.info('  Bins per observable: %s', self.n_bins_x)

        # Not yet trained
        self.n_parameters = None
        self.n_observables = None
        self.n_bins = None
        self.edges = None
        self.histos = None

    def _calculate_binning(self, theta, x, observables=None, lower_cutoff_percentile=0., upper_cutoff_percentile=100.):
        all_theta_x = np.hstack([theta, x]).T

        # Number of bins
        n_samples = x.shape[0]
        n_parameters = theta.shape[1]
        n_all_observables = x.shape[1]

        # Observables to actually use
        if observables is None:
            observables = list(range(n_all_observables))

        n_binned_observables = len(observables)

        # TODO: better automatic bin number determination
        recommended_n_bins = 10 + int(round(n_samples ** (1. / 3.), 0))
        logger.info('Recommended total number of bins: %s', recommended_n_bins)

        n_bins_per_theta = self.n_bins_theta
        if n_bins_per_theta == 'auto':
            n_bins_per_theta = max(3, int(round(recommended_n_bins ** (1. / (n_parameters + n_binned_observables)))))

        n_bins_per_x = self.n_bins_x
        if n_bins_per_x == 'auto':
            n_bins_per_x = max(3, int(round(recommended_n_bins ** (1. / (n_parameters + n_binned_observables)))))

        all_n_bins = [1 for _ in range(n_all_observables)]
        for i in observables:
            all_n_bins[i] = n_bins_per_x
        all_n_bins = [n_bins_per_theta] * n_parameters + all_n_bins

        # Find edges based on percentiles
        all_edges = []
        all_ranges = []

        for i, (data, n_bins) in enumerate(zip(all_theta_x, all_n_bins)):
            edges = np.percentile(data, np.linspace(lower_cutoff_percentile, upper_cutoff_percentile, n_bins + 1))
            range_ = (np.nanmin(data) - 0.01, np.nanmax(data) + 0.01)
            edges[0], edges[-1] = range_

            # Remove zero-width bins
            widths = np.array(list(edges[1:] - edges[:-1]) + [1.])
            edges = edges[widths > 1.e-9]

            all_n_bins[i] = len(edges) - 1
            all_edges.append(edges)
            all_ranges.append(range_)

        return all_n_bins, all_edges, all_ranges

    def fit(self, theta, x, fill_empty_bins=False):

        n_samples = x.shape[0]
        self.n_parameters = theta.shape[1]
        self.n_observables = x.shape[1]

        logger.info('Filling histogram with settings:')
        logger.info('  Samples:       %s', n_samples)
        logger.info('  Parameters:    %s', self.n_parameters)
        logger.info('  Observables:   %s', self.n_observables)
        logger.info('  No empty bins: %s', fill_empty_bins)

        # Find bins
        logger.info('Calculating binning')

        self.n_bins = []
        self.edges = []
        ranges = []

        if self.separate_1d_x_histos:
            for observable in range(self.n_observables):
                histo_n_bins, histo_edges, histo_ranges = self._calculate_binning(theta, x, [observable])

                self.n_bins.append(histo_n_bins)
                self.edges.append(histo_edges)
                ranges.append(histo_ranges)

        else:
            histo_n_bins, histo_edges, histo_ranges = self._calculate_binning(theta, x)

            self.n_bins.append(histo_n_bins)
            self.edges.append(histo_edges)
            ranges.append(histo_ranges)

        for h, (histo_n_bins, histo_edges, histo_ranges) in enumerate(zip(self.n_bins, self.edges, ranges)):
            logger.info('Histogram %s: bin edges', h + 1)
            for i, (axis_bins, axis_edges, axis_range) in enumerate(zip(histo_n_bins, histo_edges, histo_ranges)):
                if i < theta.shape[1]:
                    logger.info(
                        '  theta %s: %s bins, range %s, edges %s',
                        i + 1, axis_bins, axis_range, axis_edges
                    )
                else:
                    logger.info(
                        '  x %s:     %s bins, range %s, edges %s',
                        i + 1 - theta.shape[1], axis_bins, axis_range, axis_edges
                    )

        # Fill histograms
        logger.info('Filling histograms')
        self.histos = []
        theta_x = np.hstack([theta, x])

        for histo_edges, histo_ranges, histo_n_bins in zip(self.edges, ranges, self.n_bins):
            histo, _ = np.histogramdd(theta_x, bins=histo_edges, range=histo_ranges, normed=False, weights=None)

            # Avoid empty bins
            if fill_empty_bins:
                histo[histo <= 1.] = 1.

            # Calculate cell volumes
            original_shape = tuple(histo_n_bins)
            flat_shape = tuple([-1] + list(histo_n_bins[self.n_parameters:]))

            # Fix edges for bvolume calculation (to avoid larger volumes for more training data)
            modified_histo_edges = []
            for i in range(x.shape[1]):
                axis_edges = histo_edges[self.n_parameters + i]
                axis_edges[0] = min(np.percentile(x[:, i], 5.), axis_edges[1] - 0.01)
                axis_edges[-1] = max(np.percentile(x[:, i], 95.), axis_edges[-2] + 0.01)
                modified_histo_edges.append(axis_edges)

            bin_widths = [axis_edges[1:] - axis_edges[:-1] for axis_edges in modified_histo_edges]

            volumes = np.ones(flat_shape[1:])
            for obs in range(self.n_observables):
                # Broadcast bin widths to array with shape like volumes
                bin_widths_broadcasted = np.ones(flat_shape[1:])
                for indices in np.ndindex(flat_shape[1:]):
                    bin_widths_broadcasted[indices] = bin_widths[obs][indices[obs]]
                volumes[:] *= bin_widths_broadcasted

            # Normalize histograms (for each theta bin)
            histo = histo.reshape(flat_shape)

            for i in range(histo.shape[0]):
                histo[i] /= np.sum(histo[i])
                histo[i] /= volumes

            histo = histo.reshape(original_shape)

            # Avoid NaNs
            histo[np.invert(np.isfinite(histo))] = 0.

            self.histos.append(histo)

    def log_likelihood(self, theta, x):
        theta_x = np.hstack([theta, x])

        log_p = 0.

        for histo, histo_edges, n_bins in zip(self.histos, self.edges, self.n_bins):
            histo_indices = []

            for j in range(theta_x.shape[1]):
                indices = np.searchsorted(histo_edges[j],
                                          theta_x[:, j],
                                          side="right") - 1

                indices[indices < 0] = 0
                indices[indices >= n_bins[j]] = n_bins[j] - 1

                histo_indices.append(indices)

            log_p += np.log(histo[histo_indices])

        return log_p