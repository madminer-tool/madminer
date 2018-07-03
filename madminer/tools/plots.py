from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from matplotlib import pyplot as plt
import matplotlib


def plot_2d_morphing_basis(morpher,
                           xlabel=r'$\theta_0$',
                           ylabel=r'$\theta_1$',
                           xrange=(-1.,1.),
                           yrange=(-1.,1.),
                           crange=(1.,100.),
                           resolution=100):

    basis = morpher.basis

    assert basis is not None, "No basis defined"
    assert basis.shape[1] == 2, 'Only 2d problems can be plotted with this function'

    xi, yi = np.linspace(xrange[0], xrange[1], resolution), np.linspace(yrange[0], yrange[1], resolution)
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

    pcm = ax.pcolormesh(xi, yi, squared_weights,
                        norm=matplotlib.colors.LogNorm(vmin=crange[0], vmax=crange[1]),
                        cmap='viridis_r')
    cbar = fig.colorbar(pcm, ax=ax, extend='both')

    plt.scatter(basis[:, 0], basis[:, 1], s=50., c='black')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar.set_label(r'$\sqrt{\sum w_i^2}$')
    plt.xlim(xrange[0], xrange[1])
    plt.ylim(yrange[0], yrange[1])

    plt.tight_layout()

    return fig
