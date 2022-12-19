import logging
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def plot_fisher_information_contours_2d(
    fisher_information_matrices,
    fisher_information_covariances=None,
    reference_thetas=None,
    contour_distance=1.0,
    xlabel=r"$\theta_0$",
    ylabel=r"$\theta_1$",
    xrange=(-1.0, 1.0),
    yrange=(-1.0, 1.0),
    labels=None,
    inline_labels=None,
    resolution=500,
    colors=None,
    linestyles=None,
    linewidths=1.5,
    alphas=1.0,
    alphas_uncertainties=0.25,
    sigma_uncertainties=1,
    ax=None,
):
    """
    Visualizes 2x2 Fisher information matrices as contours of constant Fisher distance from a reference point `theta0`.

    The local (tangent-space) approximation is used: distances `d(theta)` are given by
    `d(theta)^2 = (theta - theta0)_i I_ij (theta - theta0)_j`, summing over `i` and `j`.

    Parameters
    ----------
    fisher_information_matrices : list of ndarray
        Fisher information matrices, each with shape (2,2).

    fisher_information_covariances : None or list of (ndarray or None), optional
        Covariance matrices for the Fisher information matrices. Has to have the same length as
        fisher_information_matrices, and each entry has to be None (no uncertainty) or a tensor with shape
        (2,2,2,2). Default value: None.

    reference_thetas : None or list of (ndarray or None), optional
        Reference points from which the distances are calculated. If None, the origin (0,0) is used. Default value:
        None.

    contour_distance : float, optional.
        Distance threshold at which the contours are drawn. Default value: 1.

    xlabel : str, optional
        Label for the x axis. Default value: r'$\theta_0$'.

    ylabel : str, optional
        Label for the y axis. Default value: r'$\theta_1$'.

    xrange : tuple of float, optional
        Range `(min, max)` for the x axis. Default value: (-1., 1.).

    yrange : tuple of float, optional
        Range `(min, max)` for the y axis. Default value: (-1., 1.).

    labels : None or list of (str or None), optional
        Legend labels for the contours. Default value: None.

    inline_labels : None or list of (str or None), optional
        Inline labels for the contours. Default value: None.

    resolution : int
        Number of points per axis for the calculation of the distances. Default value: 500.

    colors : None or str or list of str, optional
        Matplotlib line (and error band) colors for the contours. If None, uses default colors. Default value: None.

    linestyles : None or str or list of str, optional
        Matploitlib line styles for the contours. If None, uses default linestyles. Default value: None.

    linewidths : float or list of float, optional
        Line widths for the contours. Default value: 1.5.

    alphas : float or list of float, optional
        Opacities for the contours. Default value: 1.

    alphas_uncertainties : float or list of float, optional
        Opacities for the error bands. Default value: 0.25.

    sigma_uncertainties : float, optional
        Number of gaussian sigmas used when presenting uncertainty bands. Default value: 1.

    ax: axes or None, optional
        Predefined axes as part of figure instead of standalone figure. Default: None

    Returns
    -------
    figure : Figure
        Plot as Matplotlib Figure instance.

    """
    # Input data
    fisher_information_matrices = np.asarray(fisher_information_matrices)

    n_matrices = fisher_information_matrices.shape[0]

    if fisher_information_matrices.shape != (n_matrices, 2, 2):
        raise RuntimeError(f"Fisher information matrices have shape {fisher_information_matrices.shape}. Not (n, 2,2)!")

    if fisher_information_covariances is None:
        fisher_information_covariances = [None for _ in range(n_matrices)]

    if reference_thetas is None:
        reference_thetas = [None for _ in range(n_matrices)]

    d2_threshold = contour_distance**2.0

    # Line formatting
    if colors is None:
        colors = ["C" + str(i) for i in range(10)] * (n_matrices // 10 + 1)
    elif not isinstance(colors, list):
        colors = [colors for _ in range(n_matrices)]

    if linestyles is None:
        linestyles = ["solid", "dashed", "dotted", "dashdot"] * (n_matrices // 4 + 1)
    elif not isinstance(linestyles, list):
        linestyles = [linestyles for _ in range(n_matrices)]

    if not isinstance(linewidths, list):
        linewidths = [linewidths for _ in range(n_matrices)]

    if not isinstance(alphas, list):
        alphas = [alphas for _ in range(n_matrices)]

    if not isinstance(alphas_uncertainties, list):
        alphas_uncertainties = [alphas_uncertainties for _ in range(n_matrices)]

    # Grid
    xi = np.linspace(xrange[0], xrange[1], resolution)
    yi = np.linspace(yrange[0], yrange[1], resolution)
    xx, yy = np.meshgrid(xi, yi, indexing="xy")
    xx, yy = xx.flatten(), yy.flatten()
    thetas = np.vstack((xx, yy)).T

    # Theta from reference thetas
    d_thetas = []
    for reference_theta in reference_thetas:
        if reference_theta is None:
            d_thetas.append(thetas)
        else:
            d_thetas.append(thetas - reference_theta)
    d_thetas = np.array(d_thetas)  # Shape (n_matrices, n_thetas, n_parameters)

    # Calculate Fisher distances
    fisher_distances_squared = np.einsum("mni,mij,mnj->mn", d_thetas, fisher_information_matrices, d_thetas)
    fisher_distances_squared = fisher_distances_squared.reshape((n_matrices, resolution, resolution))

    # Calculate uncertainties of Fisher distances
    fisher_distances_squared_uncertainties = []
    for d_theta, inf_cov in zip(d_thetas, fisher_information_covariances):
        if inf_cov is None:
            fisher_distances_squared_uncertainties.append(None)
            continue

        var = np.einsum("ni,nj,ijkl,nk,nl->n", d_theta, d_theta, inf_cov, d_theta, d_theta)

        uncertainties = (var**0.5).reshape((resolution, resolution))
        fisher_distances_squared_uncertainties.append(uncertainties)

        logger.debug("Std: %s", uncertainties)

    # Plot results
    do_fig = False
    if ax is None:
        do_fig = True
        fig = plt.figure(figsize=(5.0, 5.0))
        ax = plt.gca()

    # Error bands
    for i in range(n_matrices):
        if fisher_information_covariances[i] is not None:
            d2_up = fisher_distances_squared[i] + sigma_uncertainties * fisher_distances_squared_uncertainties[i]
            d2_down = fisher_distances_squared[i] - sigma_uncertainties * fisher_distances_squared_uncertainties[i]
            band = (d2_up > d2_threshold) * (d2_down < d2_threshold) + (d2_up < d2_threshold) * (d2_down > d2_threshold)

            plt.contourf(xi, yi, band, [0.5, 2.5], colors=colors[i], alpha=alphas_uncertainties[i])

    # Predictions
    for i in range(n_matrices):
        cs = ax.contour(
            xi,
            yi,
            fisher_distances_squared[i],
            np.array([d2_threshold]),
            colors=colors[i],
            linestyles=linestyles[i],
            linewidths=linewidths[i],
            alpha=alphas[i],
            label=None if labels is None else labels[i],
        )

        if inline_labels is not None and inline_labels[i] is not None and len(inline_labels[i]) > 0:
            ax.clabel(cs, cs.levels, inline=True, fontsize=12, fmt={d2_threshold: inline_labels[i]})

    # Legend and decorations
    if labels is not None:
        ax.legend()

    if do_fig:
        plt.axes().set_xlim(xrange)
        plt.axes().set_ylim(yrange)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        return fig
    else:
        return ax


def plot_fisherinfo_barplot(
    fisher_information_matrices,
    labels,
    determinant_indices=None,
    eigenvalue_colors=None,
    bar_colors=None,
):
    """

    Parameters
    ----------
    fisher_information_matrices : list of ndarray
        Fisher information matrices

    labels : list of str
        Labels for the x axis

    determinant_indices : list of int or None, optional
        If not None, the determinants will be based only on the indices given here. Default value: None.

    eigenvalue_colors : None or list of str
        Colors for the eigenvalue decomposition. If None, default colors are used. Default value: None.

    bar_colors : None or list of str
        Colors for the determinant bars. If None, default colors are used. Default value: None.

    Returns
    -------
    figure : Figure
        Plot as Matplotlib Figure instance.

    """

    # Prepare data
    if determinant_indices is None:
        matrices_for_determinants = fisher_information_matrices
    else:
        matrices_for_determinants = [m[determinant_indices, determinant_indices] for m in fisher_information_matrices]

    size_upper = len(fisher_information_matrices[1])
    size_lower = len(matrices_for_determinants[1])
    exponent_lower = 1.0 / float(size_lower)

    determinants = [np.linalg.det(m) ** exponent_lower for m in matrices_for_determinants]

    assert len(determinants) == len(labels)
    n_entries = len(determinants)

    # Calculate eigenvalues + eigenvalue composition
    eigenvalues = []
    eigenvalues_dominant_components = []
    eigenvalues_composition = []

    for m in fisher_information_matrices:
        v, w = np.linalg.eig(m)
        w = np.transpose(w)
        v, w = zip(*sorted(zip(v, w), key=lambda x: x[0], reverse=True))
        temp = []
        temp_dominant_components = []
        temp_composition = []
        for vi, wi in zip(v, w):
            temp.append(vi)
            temp_dominant_components.append(np.argmax(np.absolute(wi)))
            temp_composition.append(wi * wi / (sum(wi * wi)))

        eigenvalues.append(temp)
        eigenvalues_dominant_components.append(temp_dominant_components)
        eigenvalues_composition.append(temp_composition)

    # x positioning
    base_xvalues = np.linspace(0.0, float(n_entries) - 1.0, n_entries)
    base_xmin = base_xvalues[0]
    base_xmax = base_xvalues[n_entries - 1] + 1.0
    xmin_eigenvalues = base_xvalues + 0.08
    xmax_eigenvalues = base_xvalues + 0.92
    xpos_ticks = base_xvalues + 0.5
    xpos_lower = base_xvalues + 0.5
    width_lower = 0.8

    # Colors
    if bar_colors is None:
        bar_colors = ["0.5" for _ in range(n_entries)]
        bar_colors_light = ["0.9" for _ in range(n_entries)]
    else:
        bar_colors_light = bar_colors

    if eigenvalue_colors is None:
        eigenvalue_colors = [f"C{i}" for i in range(10)]
    eigenvalue_linewidth = 1.5

    # Upper plot
    fig = plt.figure(figsize=(10.0, 7.0))
    ax1 = plt.subplot(211)

    # Plot eigenvalues
    for i in range(n_entries):
        for eigenvalue, composition in zip(eigenvalues[i], eigenvalues_composition[i]):
            # Gap sizing
            n_gaps = -1
            minimal_fraction_for_plot = 0.01
            for fraction in composition:
                if fraction >= minimal_fraction_for_plot:
                    n_gaps += 1
            gap_fraction = 0.04
            gap_correction_factor = 1.0 - n_gaps * gap_fraction

            fraction_finished = 0.0

            for component, fraction in enumerate(composition):
                if fraction >= minimal_fraction_for_plot:
                    plt.hlines(
                        [eigenvalue],
                        xmin_eigenvalues[i] + fraction_finished * (xmax_eigenvalues[i] - xmin_eigenvalues[i]),
                        xmin_eigenvalues[i]
                        + (fraction_finished + gap_correction_factor * fraction)
                        * (xmax_eigenvalues[i] - xmin_eigenvalues[i]),
                        eigenvalue_colors[component],
                        linestyles="solid",
                        linewidth=eigenvalue_linewidth,
                    )
                    fraction_finished += gap_correction_factor * fraction + gap_fraction

    ax1.set_yscale("log")
    ax1.set_xlim([base_xmin - 0.2, base_xmax + 0.2])
    y_max = max([max(ev) for ev in eigenvalues])
    ax1.set_ylim(0.0001 * y_max, 2.0 * y_max)

    ax1.set_xticks(xpos_ticks)
    ax1.set_xticklabels(["" for _ in labels], rotation=40, ha="right")
    ax1.set_ylabel(r"$I_{ij}$ eigenvalues")

    # Lower plot
    ax3 = plt.subplot(212)

    bar_plot = ax3.bar(xpos_lower, determinants, width=width_lower, log=False)

    for i in range(n_entries):
        bar_plot[i].set_color(bar_colors_light[i])
        bar_plot[i].set_edgecolor(bar_colors[i])

    ax3.set_xlim([base_xmin - 0.2, base_xmax + 0.2])
    ax3.set_ylim([0.0, max(determinants) * 1.05])

    ax3.set_xticks(xpos_ticks)
    ax3.set_xticklabels(labels, rotation=40, ha="right")
    ax3.set_ylabel(r"$(\det \ I_{ij})^{1/" + str(size_lower) + r"}$")

    plt.tight_layout()
    return fig


def plot_distribution_of_information(
    xbins,
    xsecs,
    fisher_information_matrices,
    fisher_information_matrices_aux=None,
    xlabel=None,
    xmin=None,
    xmax=None,
    log_xsec=False,
    norm_xsec=True,
    epsilon=1.0e-9,
    figsize=(5.4, 4.5),
    fontsize=None,
):
    """
    Plots the distribution of the cross section together with the distribution of the Fisher information.

    Parameters
    ----------
    xbins : list of float
        Bin boundaries.

    xsecs : list of float
        Cross sections (in pb) per bin.

    fisher_information_matrices : list of ndarray
        Fisher information matrices for each bin.

    fisher_information_matrices_aux : list of ndarray or None, optional
        Additional Fisher information matrices for each bin (will be plotted with a dashed line).

    xlabel : str or None, optional
        Label for the x axis.

    xmin : float or None, optional
        Minimum value for the x axis.

    xmax : float or None, optional
        Maximum value for the x axis.

    log_xsec : bool, optional
        Whether to plot the cross section on a logarithmic y axis.

    norm_xsec : bool, optional
        Whether the cross sections are normalized to 1.

    epsilon : float, optional
        Numerical factor.

    figsize : tuple of float, optional
        Figure size, default: (5.4, 4.5)

    fontsize: float, optional
        Fontsize, default None

    Returns
    -------
    figure : Figure
        Plot as Matplotlib Figure instance.

    """
    # prepare Plot
    if fontsize is not None:
        matplotlib.rcParams.update({"font.size": fontsize})

    # Prepare data
    n_entries = len(fisher_information_matrices)
    size = len(fisher_information_matrices[1])
    exponent = 1.0 / float(size)

    determinants = [np.nan_to_num(np.linalg.det(m) ** exponent) for m in fisher_information_matrices]

    if fisher_information_matrices_aux is not None:
        determinants_aux = [np.nan_to_num(np.linalg.det(m) ** exponent) for m in fisher_information_matrices_aux]

    if xlabel is None:
        xlabel = ""

    # Normalize xsecs
    if norm_xsec:
        norm = 1.0 / max(sum([xs for xs in xsecs]), epsilon)
    else:
        norm = 1.0
    xsec_norm = [norm * xs for xs in xsecs]

    # Get xvals from xbins
    xvals = [(xbins[i] + xbins[i + 1]) / 2 for i in range(0, len(xbins) - 1)]
    xvals = [xbins[0] - epsilon] + xvals + [xbins[len(xbins) - 1] + epsilon]
    assert len(xvals) == n_entries

    # Plotting options
    xs_color = "black"
    xs_linestyle = "-"
    xs_linewidth = 1.5

    det_color = "red"
    det_linestyle = "-"
    det_linewidth = 1.5
    det_fill_alpha = 0.1

    det_aux_color = "red"
    det_aux_linestyle = "--"
    det_aux_linewidth = 1.5

    # xsec plot
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(111)
    # fig.subplots_adjust(left=0.1667, right=0.8333, bottom=0.17, top=0.97)

    if log_xsec:
        ax1.set_yscale("log")

    ax1.hist(
        xvals,
        weights=xsec_norm,
        bins=xbins,
        range=(xmin, xmax),
        histtype="step",
        color=xs_color,
        linewidth=xs_linewidth,
        linestyle=xs_linestyle,
    )

    if norm_xsec:
        ax1.set_ylabel(r"Normalized distribution", color=xs_color)
    else:
        ax1.set_ylabel(r"$\sigma$ [pb/bin]")
    ax1.set_xlim([xmin, xmax])
    ax1.set_ylim([0.0, max(xsec_norm) * 1.05])
    ax1.set_xlabel(xlabel)
    for tl in ax1.get_yticklabels():
        tl.set_color(xs_color)

    # det plot
    ax2 = ax1.twinx()

    if fisher_information_matrices_aux is not None:
        ax2.hist(
            xvals,
            weights=determinants_aux,
            bins=xbins,
            range=(xmin, xmax),
            histtype="step",
            color=det_aux_color,
            linewidth=det_aux_linewidth,
            linestyle=det_aux_linestyle,
        )

    ax2.hist(
        xvals,
        weights=determinants,
        bins=xbins,
        range=(xmin, xmax),
        histtype="stepfilled",
        alpha=det_fill_alpha,
        color=det_color,
        linewidth=0.0,
    )

    ax2.hist(
        xvals,
        weights=determinants,
        bins=xbins,
        range=(xmin, xmax),
        histtype="step",
        color=det_color,
        linewidth=det_linewidth,
        linestyle=det_linestyle,
    )

    ax2.set_xlim([xmin, xmax])
    ax2.set_ylim([0.0, max(determinants) * 1.1])
    ax2.set_ylabel(r"$(\det \; I_{ij})^{1/" + str(size) + "}$", color=det_color)
    for tl in ax2.get_yticklabels():
        tl.set_color(det_color)

    return fig
