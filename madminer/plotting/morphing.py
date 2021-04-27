import logging
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def plot_1d_morphing_basis(morpher, xlabel=r"$\theta$", xrange=(-1.0, 1.0), resolution=100):
    """
    Visualizes a morphing basis and morphing errors for problems with a two-dimensional parameter space.

    Parameters
    ----------
    morpher : PhysicsMorpher
        PhysicsMorpher instance with defined basis.

    xlabel : str, optional
        Label for the x axis. Default value: r'$\theta$'.

    xrange : tuple of float, optional
        Range `(min, max)` for the x axis. Default value: (-1., 1.).

    resolution : int, optional
        Number of points per axis for the rendering of the squared morphing weights. Default value: 100.

    Returns
    -------
    figure : Figure
        Plot as Matplotlib Figure instance.

    """

    basis = morpher.basis

    assert basis is not None, "No basis defined"
    assert basis.shape[1] == 1, "Only 1d problems can be plotted with this function"

    theta_test = np.linspace(xrange[0], xrange[1], resolution).reshape((-1, 1))

    squared_weights = []
    for theta in theta_test:
        wi = morpher.calculate_morphing_weights(theta, None)
        squared_weights.append(np.sum(wi * wi) ** 0.5)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    ax.plot(theta_test.flatten(), squared_weights)

    plt.scatter(basis[:, 0], [0.0 for _ in basis[:, 0]], s=50.0, c="black")

    plt.xlabel(xlabel)
    plt.ylabel(r"$\sqrt{\sum w_i^2}$")
    plt.xlim(xrange[0], xrange[1])
    plt.ylim(0.0, None)

    plt.tight_layout()

    return fig


def plot_2d_morphing_basis(
    morpher,
    xlabel=r"$\theta_0$",
    ylabel=r"$\theta_1$",
    xrange=(-1.0, 1.0),
    yrange=(-1.0, 1.0),
    crange=(1.0, 100.0),
    resolution=100,
):
    """
    Visualizes a morphing basis and morphing errors for problems with a two-dimensional parameter space.

    Parameters
    ----------
    morpher : PhysicsMorpher
        PhysicsMorpher instance with defined basis.

    xlabel : str, optional
        Label for the x axis. Default value: r'$\theta_0$'.

    ylabel : str, optional
        Label for the y axis. Default value: r'$\theta_1$'.

    xrange : tuple of float, optional
        Range `(min, max)` for the x axis. Default value: (-1., 1.).

    yrange : tuple of float, optional
        Range `(min, max)` for the y axis. Default value: (-1., 1.).

    crange : tuple of float, optional
        Range `(min, max)` for the color map. Default value: (1., 100.).

    resolution : int, optional
        Number of points per axis for the rendering of the squared morphing weights. Default value: 100.

    Returns
    -------
    figure : Figure
        Plot as Matplotlib Figure instance.

    """

    basis = morpher.basis

    assert basis is not None, "No basis defined"
    assert basis.shape[1] == 2, "Only 2d problems can be plotted with this function"

    xi, yi = (np.linspace(xrange[0], xrange[1], resolution), np.linspace(yrange[0], yrange[1], resolution))
    xx, yy = np.meshgrid(xi, yi)
    xx, yy = xx.reshape((-1, 1)), yy.reshape((-1, 1))
    theta_test = np.hstack([xx, yy])

    squared_weights = []
    for theta in theta_test:
        wi = morpher.calculate_morphing_weights(theta, None)
        squared_weights.append(np.sum(wi * wi) ** 0.5)
    squared_weights = np.array(squared_weights).reshape((resolution, resolution))

    fig = plt.figure(figsize=(6.5, 5))
    ax = plt.gca()

    pcm = ax.pcolormesh(
        xi, yi, squared_weights, norm=matplotlib.colors.LogNorm(vmin=crange[0], vmax=crange[1]), cmap="viridis_r"
    )
    cbar = fig.colorbar(pcm, ax=ax, extend="both")

    plt.scatter(basis[:, 0], basis[:, 1], s=50.0, c="black")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar.set_label(r"$\sqrt{\sum w_i^2}$")
    plt.xlim(xrange[0], xrange[1])
    plt.ylim(yrange[0], yrange[1])

    plt.tight_layout()

    return fig


def plot_nd_morphing_basis_scatter(morpher, crange=(1.0, 100.0), n_test_thetas=1000):
    """
    Visualizes a morphing basis and morphing errors with scatter plots between each pair of operators.

    Parameters
    ----------
    morpher : PhysicsMorpher
        PhysicsMorpher instance with defined basis.

    crange : tuple of float, optional
        Range `(min, max)` for the color map. Default value: (1. 100.).

    n_test_thetas : int, optional
        Number of random points evaluated. Default value: 1000.

    Returns
    -------
    figure : Figure
        Plot as Matplotlib Figure instance.

    """
    basis = morpher.basis

    assert basis is not None, "No basis defined"

    n_parameters = basis.shape[1]

    #  Get squared weights
    thetas, squared_weights = morpher.evaluate_morphing(n_test_thetas=n_test_thetas, return_weights_and_thetas=True)

    # Plot
    fig = plt.figure(figsize=((n_parameters - 1) * 5.0, (n_parameters - 1) * 4.0))

    for iy in range(1, n_parameters):
        for ix in range(0, iy):
            i_panel = 1 + (iy - 1) * (n_parameters - 1) + ix
            ax = plt.subplot(n_parameters - 1, n_parameters - 1, i_panel)

            sc = plt.scatter(
                thetas[:, ix],
                thetas[:, iy],
                c=squared_weights,
                s=20.0,
                norm=matplotlib.colors.LogNorm(vmin=crange[0], vmax=crange[1]),
                cmap="viridis_r",
            )
            cbar = fig.colorbar(sc, ax=ax, extend="both")

            plt.scatter(basis[:, ix], basis[:, iy], s=100.0, lw=1.0, edgecolor="black", c="white")

            plt.xlabel(r"$\theta_" + str(ix) + "$")
            plt.ylabel(r"$\theta_" + str(iy) + "$")
            cbar.set_label(r"$\sqrt{\sum w_i^2}$")

    plt.tight_layout()

    return fig


def plot_nd_morphing_basis_slices(morpher, crange=(1.0, 100.0), resolution=50):
    """
    Visualizes a morphing basis and morphing errors with two-dimensional slices through parameter space.

    Parameters
    ----------
    morpher : PhysicsMorpher
        PhysicsMorpher instance with defined basis.

    crange : tuple of float, optional
        Range `(min, max)` for the color map.

    resolution : int, optional
        Number of points per panel and axis for the rendering of the squared morphing weights. Default value: 50.

    Returns
    -------
    figure : Figure
        Plot as Matplotlib Figure instance.

    """
    basis = morpher.basis

    assert basis is not None, "No basis defined"

    n_parameters = basis.shape[1]

    # Plot
    fig = plt.figure(figsize=((n_parameters - 1) * 5.0, (n_parameters - 1) * 4.0))

    for iy in range(1, n_parameters):
        for ix in range(0, iy):
            i_panel = 1 + (iy - 1) * (n_parameters - 1) + ix
            ax = plt.subplot(n_parameters - 1, n_parameters - 1, i_panel)

            # Grid
            xrange = morpher.parameter_range[ix]
            yrange = morpher.parameter_range[iy]
            xi = np.linspace(xrange[0], xrange[1], resolution)
            yi = np.linspace(yrange[0], yrange[1], resolution)
            xx, yy = np.meshgrid(xi, yi)
            xx, yy = xx.flatten(), yy.flatten()

            theta_test = np.zeros((resolution ** 2, n_parameters))
            theta_test[:, ix] = xx
            theta_test[:, iy] = yy

            # Get squared weights
            squared_weights = []
            for theta in theta_test:
                wi = morpher.calculate_morphing_weights(theta, None)
                squared_weights.append(np.sum(wi * wi) ** 0.5)
            squared_weights = np.array(squared_weights).reshape((resolution, resolution))

            pcm = ax.pcolormesh(
                xi,
                yi,
                squared_weights,
                norm=matplotlib.colors.LogNorm(vmin=crange[0], vmax=crange[1]),
                cmap="viridis_r",
            )
            cbar = fig.colorbar(pcm, ax=ax, extend="both")

            plt.scatter(basis[:, ix], basis[:, iy], s=100.0, lw=1.0, edgecolor="black", c="white")

            plt.xlabel(r"$\theta_" + str(ix) + "$")
            plt.ylabel(r"$\theta_" + str(iy) + "$")
            cbar.set_label(r"$\sqrt{\sum w_i^2}$")

    plt.tight_layout()

    return fig
