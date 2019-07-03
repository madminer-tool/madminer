from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import logging
from madminer.utils.various import weighted_quantile

logger = logging.getLogger(__name__)


class Histo:
    def __init__(self, x, weights=None, bins=20, fill_empty=None):
        """
        Initialize and fit an n-dim histogram.

        Parameters
        ----------
        x : ndarray
            Data with shape (n_events, n_observables)

        weights : None or ndarray, optional
            Weights with shape (n_events,)

        bins : int or list of int or list of ndarray
            Number of bins per observable (when int or list of int), or actual bin boundaries (when list of ndarray)

        fill_empty : None or float
            If not None, this number is added to all empty bins

        """

        # Data
        if len(x.shape) == 1:
            x = x.reshape((-1, 1))
        self.n_samples, self.n_observables = x.shape

        if weights is not None:
            weights = weights.flatten()
            assert weights.shape == (self.n_samples,), "Inconsistent weight shape {} should be {}".format(
                weights.shape, (self.n_samples,)
            )

        logger.debug("Creating histogram:")
        logger.debug("  Samples:       %s", self.n_samples)
        logger.debug("  Observables:   %s with means %s", self.n_observables, np.mean(x, axis=0))
        logger.debug("  Weights:       %s", weights is not None)

        # Calculate binning
        self.n_bins, self.edges = self._calculate_binning(x, bins, weights=weights)

        logger.debug("Binning:")
        for i, (n_bins, edges) in enumerate(zip(self.n_bins, self.edges)):
            logger.debug("  Observable %s: %s bins with edges %s", i + 1, n_bins, edges)

        # Fill histogram
        self.histo = self._fit(x, weights, fill_empty)

    def _calculate_binning(self, x, bins_in, weights=None):
        if isinstance(bins_in, int):
            bins_in = [bins_in for _ in range(self.n_observables)]

        # Find binning along each observable direction
        n_bins = []
        bin_edges = []

        for this_bins, this_x in zip(bins_in, x.T):
            if isinstance(this_bins, int):
                bin_edges.append(self._adaptive_binning(this_x, this_bins, weights=weights))
            else:
                bin_edges.append(this_bins)
            n_bins.append(len(bin_edges[-1]) - 1)

        return n_bins, bin_edges

    def _adaptive_binning(self, x, n_bins, weights=None, lower_cutoff_percentile=0.0, upper_cutoff_percentile=100.0):
        edges = weighted_quantile(
            x,
            quantiles=np.linspace(lower_cutoff_percentile / 100.0, upper_cutoff_percentile / 100.0, n_bins + 1),
            sample_weight=weights,
            old_style=True,
        )
        range_ = (np.nanmin(x) - 0.01, np.nanmax(x) + 0.01)
        edges[0], edges[-1] = range_

        # Remove zero-width bins
        widths = np.array(list(edges[1:] - edges[:-1]) + [1.0])
        edges = edges[widths > 1.0e-9]

        return edges

    def _fit(self, x, weights=None, fill_empty=None):

        # Fill histograms
        ranges = [(edges[0], edges[-1]) for edges in self.edges]
        histo, _ = np.histogramdd(x, bins=self.edges, range=ranges, normed=False, weights=weights)

        # Avoid empty bins
        if fill_empty is not None:
            histo[histo <= fill_empty] = fill_empty

        # Fix edges for bvolume calculation (to avoid larger volumes for more training data)
        modified_histo_edges = []
        for i in range(x.shape[1]):
            axis_edges = self.edges[i]
            axis_edges[0] = min(np.percentile(x[:, i], 5.0), axis_edges[1] - 0.01)
            axis_edges[-1] = max(np.percentile(x[:, i], 95.0), axis_edges[-2] + 0.01)
            modified_histo_edges.append(axis_edges)

        # Calculate cell volumes
        bin_widths = [axis_edges[1:] - axis_edges[:-1] for axis_edges in modified_histo_edges]

        shape = tuple(self.n_bins)
        volumes = np.ones(shape)
        for obs in range(self.n_observables):
            # Broadcast bin widths to array with shape like volumes
            bin_widths_broadcasted = np.ones(shape)
            for indices in np.ndindex(shape):
                bin_widths_broadcasted[indices] = bin_widths[obs][indices[obs]]
            volumes[:] *= bin_widths_broadcasted

        # Normalize histograms (for each theta bin)
        histo /= np.sum(histo)
        histo /= volumes

        # Avoid NaNs
        histo[np.invert(np.isfinite(histo))] = 0.0

        return histo

    def log_likelihood(self, x):
        """
        Calculates the log likelihood with the histogram.

        Parameters
        ----------
        x : ndarray
            Data with shape (n_eval, n_observables)

        Returns
        -------
        log_likelihood : float
            Log likelihood.

        """

        if len(x.shape) == 1:
            x = x.reshape((-1, 1))
        assert x.shape[1] == self.n_observables

        # Find hist indices
        all_indices = []
        for i in range(self.n_observables):
            indices = np.searchsorted(self.edges[i], x[:, i], side="right") - 1
            indices[indices < 0] = 0
            indices[indices >= self.n_bins[i]] = self.n_bins[i] - 1
            all_indices.append(indices)

        # Return log likelihood
        return np.log(self.histo[tuple(all_indices)])
