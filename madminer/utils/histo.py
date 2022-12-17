import logging
import numpy as np

from madminer.utils.various import weighted_quantile

logger = logging.getLogger(__name__)


class Histo:
    def __init__(self, x, weights=None, bins=20, epsilon=0.0):
        """
        Initialize and fit an n-dim histogram.

        Parameters
        ----------
        x : ndarray
            Data with shape (n_events, n_observables)

        weights : None or ndarray, optional
            Weights with shape (n_events,). Default: None.

        bins : int or list of int or list of ndarray, optional
            Number of bins per observable (when int or list of int), or actual bin boundaries (when list of ndarray).
            Default: None.

        epsilon : float, optional
            Small number added to all bin contents. Default value: 0.

        """

        # Data
        if len(x.shape) == 1:
            x = x.reshape((-1, 1))
        self.n_samples, self.n_observables = x.shape

        if weights is not None:
            weights = weights.flatten()
            assert weights.shape == (
                self.n_samples,
            ), f"Inconsistent weight shape {weights.shape} should be {(self.n_samples,)}"

        logger.debug("Creating histogram:")
        logger.debug("  Samples:       %s", self.n_samples)
        logger.debug("  Observables:   %s with means %s", self.n_observables, np.mean(x, axis=0))
        logger.debug("  Weights:       %s", weights is not None)

        # Calculate binning
        self.n_bins, self.edges = self._calculate_binning(x, bins, weights=weights)
        self._report_binning()

        # Fill histogram
        self.histo, self.histo_uncertainties = self._fit(x, weights, epsilon)
        self._report_uncertainties()

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

    def _calculate_binning(self, x, bins_in, weights=None):
        if isinstance(bins_in, int):
            bins_in = [bins_in for _ in range(self.n_observables)]

        # Find binning along each observable direction
        num_bins = []
        bin_edges = []

        for this_bins, this_x in zip(bins_in, x.T):
            if isinstance(this_bins, int):
                edges = self._adaptive_binning(this_x, this_bins, weights=weights)
            else:
                edges = this_bins

            num_bins.append(len(edges) - 1)
            bin_edges.append(edges)

        return num_bins, bin_edges

    @staticmethod
    def _adaptive_binning(x, n_bins, weights=None, lower_cutoff_percentile=0.1, upper_cutoff_percentile=99.9):
        edges = weighted_quantile(
            x,
            quantiles=np.linspace(lower_cutoff_percentile / 100.0, upper_cutoff_percentile / 100.0, n_bins + 1),
            sample_weight=weights,
            old_style=True,
        )

        # Increase range by some safety margin
        # range_ = (np.nanmin(x) - 0.5 * (edges[1] - edges[0]), np.nanmax(x) + 0.5 * (edges[-1] - edges[-2]))
        # logger.debug("Increasing histogram range from %s to %s", (edges[0], edges[-1]), range_)
        # edges[0], edges[-1] = range_

        # Remove zero-width bins
        widths = np.array(list(edges[1:] - edges[:-1]) + [1.0])
        edges = edges[widths > 1.0e-9]

        return edges

    def _fit(self, x, weights=None, epsilon=0.0):
        # Fill histograms
        ranges = [(edges[0], edges[-1]) for edges in self.edges]

        histo, _ = np.histogramdd(
            x,
            bins=self.edges,
            range=ranges,
            normed=False,
            weights=weights,
        )
        histo_w2, _ = np.histogramdd(
            x,
            bins=self.edges,
            range=ranges,
            normed=False,
            weights=None if weights is None else weights**2,
        )

        # Uncertainties
        histo_uncertainties = histo_w2**0.5

        # Normalize histograms to sum to 1
        histo_uncertainties /= np.sum(histo)
        histo /= np.sum(histo)

        # Avoid empty bins, and normalize again
        histo[:] += epsilon
        histo_uncertainties[:] += epsilon
        histo_uncertainties /= np.sum(histo)
        histo /= np.sum(histo)

        # Calculate cell volumes
        # Fix edges for bvolume calculation (to avoid larger volumes for more training data)
        modified_histo_edges = []
        for i in range(x.shape[1]):
            axis_edges = np.copy(self.edges[i])
            if len(axis_edges) > 2:
                axis_edges[0] = max(
                    axis_edges[0], axis_edges[1] - 2.0 * (axis_edges[2] - axis_edges[1])
                )  # First bin is treated as at most twice as big as second
                axis_edges[-1] = min(
                    axis_edges[-1], axis_edges[-1] + 2.0 * (axis_edges[-1] - axis_edges[-2])
                )  # Last bin is treated as at most twice as big as second-to-last
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

        # Normalize histogram bins to volume
        histo_uncertainties /= volumes
        histo /= volumes

        # Avoid NaNs
        histo_uncertainties[np.invert(np.isfinite(histo))] = 1.0e9
        histo_uncertainties[np.invert(np.isfinite(histo_uncertainties))] = 0.0
        histo[np.invert(np.isfinite(histo))] = 0.0

        return histo, histo_uncertainties

    def _report_binning(self):
        logger.debug("Binning:")
        for i, (n_bins, edges) in enumerate(zip(self.n_bins, self.edges), start=1):
            logger.debug("  Observable %s: %s bins with edges %s", i, n_bins, edges)

    def _report_uncertainties(self):
        rel_uncertainties = np.where(
            self.histo.flatten() > 0.0,
            self.histo_uncertainties.flatten() / self.histo.flatten(),
            np.nan,
        )
        if np.nanmax(rel_uncertainties) > 0.5:
            logger.debug(
                "Large statistical uncertainties in histogram! Relative uncertainties range from %.0f%% to %.0f%% "
                "with median %.0f%%.",
                100.0 * np.nanmin(rel_uncertainties),
                100.0 * np.nanmax(rel_uncertainties),
                100.0 * np.nanmedian(rel_uncertainties),
            )

        logger.debug("Statistical uncertainties in histogram:")
        for i, (histo, unc, rel_unc) in enumerate(
            zip(self.histo.flatten(), self.histo_uncertainties.flatten(), rel_uncertainties)
        ):
            logger.debug("  Bin %s: %.5f +/- %.5f (%.0f%%)", i + 1, histo, unc, 100.0 * rel_unc)
