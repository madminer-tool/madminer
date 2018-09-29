from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from madminer.utils.various import create_missing_folders

def plot_2d_morphing_basis(morpher,
                           xlabel=r'$\theta_0$',
                           ylabel=r'$\theta_1$',
                           xrange=(-1., 1.),
                           yrange=(-1., 1.),
                           crange=(1., 100.),
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


def plot_nd_morphing_basis_scatter(morpher,
                                   crange=(1., 100.),
                                   n_test_thetas=1000):
    basis = morpher.basis

    assert basis is not None, "No basis defined"

    n_parameters = basis.shape[1]

    #  Get squared weights
    thetas, squared_weights = morpher.evaluate_morphing(n_test_thetas=n_test_thetas, return_weights_and_thetas=True)

    # Plot
    fig = plt.figure(figsize=((n_parameters - 1) * 5., (n_parameters - 1) * 4.))

    for iy in range(1, n_parameters):
        for ix in range(0, iy):
            i_panel = 1 + (iy - 1) * (n_parameters - 1) + ix
            ax = plt.subplot(n_parameters - 1, n_parameters - 1, i_panel)

            sc = plt.scatter(thetas[:, ix], thetas[:, iy], c=squared_weights, s=20.,
                             norm=matplotlib.colors.LogNorm(vmin=crange[0], vmax=crange[1]),
                             cmap='viridis_r')
            cbar = fig.colorbar(sc, ax=ax, extend='both')

            plt.scatter(basis[:, ix], basis[:, iy], s=100., lw=1., edgecolor='black', c='white')

            plt.xlabel(r'$\theta_' + str(ix) + '$')
            plt.ylabel(r'$\theta_' + str(iy) + '$')
            cbar.set_label(r'$\sqrt{\sum w_i^2}$')

    plt.tight_layout()

    return fig


def plot_nd_morphing_basis_slices(morpher,
                                  crange=(1., 100.),
                                  resolution=50):
    basis = morpher.basis

    assert basis is not None, "No basis defined"

    n_parameters = basis.shape[1]

    # Plot
    fig = plt.figure(figsize=((n_parameters - 1) * 5., (n_parameters - 1) * 4.))

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

            theta_test = np.zeros((resolution**2, n_parameters))
            theta_test[:, ix] = xx
            theta_test[:, iy] = yy

            # Get squared weights
            squared_weights = []
            for theta in theta_test:
                wi = morpher.calculate_morphing_weights(theta, None)
                squared_weights.append(np.sum(wi * wi) ** 0.5)
            squared_weights = np.array(squared_weights).reshape((resolution, resolution))

            pcm = ax.pcolormesh(xi, yi, squared_weights,
                                norm=matplotlib.colors.LogNorm(vmin=crange[0], vmax=crange[1]),
                                cmap='viridis_r')
            cbar = fig.colorbar(pcm, ax=ax, extend='both')

            plt.scatter(basis[:, ix], basis[:, iy], s=100., lw=1., edgecolor='black', c='white')

            plt.xlabel(r'$\theta_' + str(ix) + '$')
            plt.ylabel(r'$\theta_' + str(iy) + '$')
            cbar.set_label(r'$\sqrt{\sum w_i^2}$')

    plt.tight_layout()

    return fig



def plot_fisherinfo_barplot(matrices,
                        matrices_for_determinants,
                        labels,
                        categories,
                        operatorlabels,
                        filename,
                        additional_label='',
                        top_label='',
                        normalise_determinants=False,
                        use_bar_colors=False,
                        eigenvalue_operator_legend=True
                        ):
    """
    :matrices: list (length N) of fisher infos (n x n tensors) for eigenvalue decomposition
    :matrices_for_determinants: list (length N) of fisher infos (n x n tensors) for determinant evaluation
    :labels: list (length N) of analysis label (string)
    :categories: group into categories (integer) - there will be extra space between categories
    :operatorlabels: list (length M) of operator names (string)
    :filename: save files under path (string)
    :additional_label: label (string) in lower panel
    :top_label: label (string) above top panel
    :normalise_determinants: are determinants normalized to unity (bool)
    :use_bar_colors: are bars in lower panel colored (bool)
    :eigenvalue_operator_legend: plot legend for operators (bool)
    """
    
    #################################################################################
    # Data
    #################################################################################
    
    # dimensionality of matrices
    size_upper = len(matrices[1])
    size_lower = len(matrices_for_determinants[1])
    exponent_upper = 1./float(size_upper)
    exponent_lower = 1./float(size_lower)
    
    # calculate + normalize determinants
    determinants = [np.linalg.det(m)**exponent_lower for m in matrices_for_determinants]
    
    if normalise_determinants:
        max_information = max(determinants)
        determinants = determinants / max_information
    else:
        max_information = 1.

    assert len(determinants) == len(labels)
    n_entries = len(determinants)

    print ('')
    for l,d in zip(labels,determinants):
        print (l + ': det I_{ij} =', d)
    
    #calculate eigenvalues + eigenvalue composition
    eigenvalues = []
    eigenvalues_dominant_components = []
    eigenvalues_composition = []
    
    for m in matrices:
        v,w = np.linalg.eig(m)
        w=np.transpose(w)
        v,w = zip(*sorted(zip(v,w),
                  key=lambda x: x[0],
                  reverse=True))
        temp = []
        temp_dominant_components = []
        temp_composition = []
        for vi,wi in zip(v,w):
            temp.append(vi)
            temp_dominant_components.append(np.argmax(np.absolute(wi)))
            temp_composition.append(wi * wi / (sum(wi * wi)) )
                          
        eigenvalues.append(temp)
        eigenvalues_dominant_components.append(temp_dominant_components)
        eigenvalues_composition.append(temp_composition)

    # assign categories, if they are not defined yet
    if len(categories) == 0:
        categories = [0 for i in range(n_entries)]
    
    #################################################################################
    # Plotting options
    #################################################################################
    
    # Base x values
    base_xvalues = np.linspace(0., float(n_entries) - 1., n_entries)
    for i in range(n_entries):
        base_xvalues[i] += (float(categories[i]) * 1.)
    
    base_xmin = base_xvalues[0]
    base_xmax =  base_xvalues[n_entries-1] + 1.
    
    xpos = base_xvalues + 0.2
    width = 0.6
    xmin_eigenvalues = base_xvalues + 0.08
    xmax_eigenvalues = base_xvalues + 0.92
    xpos_ticks = base_xvalues + 0.5
    
    xpos_lower = base_xvalues + 0.5
    width_lower = 0.8
    
    # barcolored - either colored or gray
    if use_bar_colors:
        bar_colors = ['red', 'blue', 'green', 'darkorange', 'fuchsia', 'turquoise']*5
        bar_colors_light = ['red', 'blue', 'green', 'darkorange', 'fuchsia', 'turquoise']*5
    else:
        bar_colors = ['0.5']*30
        bar_colors_light = ['0.9']*30

    eigenvalue_colors = ['red', 'blue', 'green', 'darkorange', 'fuchsia', 'turquoise']*5
    operator_order = [i for i in range(0,size_upper)]
    eigenvalue_linewidth = 1.5
    
    #################################################################################
    # Upper plot
    #################################################################################
    
    # Plot bars!
    fig = plt.figure(figsize=(9.,6.))
    ax1 = plt.subplot(211)
    ax1.set_yscale('log')
    fig.subplots_adjust(left=0.075,right=0.925,bottom=0.15,top=0.95,wspace=0,hspace=0)
    
    # Plot eigenvalues
    for i in range(n_entries):
        
        for eigenvalue, composition in zip(eigenvalues[i],eigenvalues_composition[i]):
            # gap sizing
            n_gaps = -1
            minimal_fraction_for_plot = 0.01
            for fraction in composition:
                if fraction >= minimal_fraction_for_plot:
                    n_gaps += 1
            gap_fraction = 0.04
            gap_correction_factor = 1. - n_gaps * gap_fraction
            
            fraction_finished = 0.
            
            for j in range(len(composition)):
                component = operator_order[j]
                fraction = composition[component]
                
                if fraction >= minimal_fraction_for_plot:
                    plt.hlines([eigenvalue],
                               xmin_eigenvalues[i] + fraction_finished * (xmax_eigenvalues[i] - xmin_eigenvalues[i]),
                               xmin_eigenvalues[i] + (fraction_finished + gap_correction_factor * fraction) * (xmax_eigenvalues[i] - xmin_eigenvalues[i]),
                               eigenvalue_colors[component],
                               linestyles='solid',
                               linewidth=eigenvalue_linewidth)
                    fraction_finished += gap_correction_factor * fraction + gap_fraction

    ax1.set_xlim([base_xmin - 0.2,base_xmax + 0.2])

    if size_upper > 2:
        orderofmagnitudes = 1.e6
        topfactor = 10.
    else:
        orderofmagnitudes = 1.e3
        topfactor = 5.

    ax1.set_ylim([ max([max(ev) for ev in eigenvalues]) / orderofmagnitudes,
              max([max(ev) for ev in eigenvalues])*topfactor])
    legend_position = max([max(ev) for ev in eigenvalues])*topfactor / (topfactor * orderofmagnitudes)**0.1

    ax1.set_xticks(xpos_ticks)
    ax1.set_xticklabels(['' for l in labels], rotation=40, ha='right')
    ax1.set_ylabel(r'$I_{ij}$ eigenvalues')
    ax1.yaxis.set_label_coords(-0.055,0.5)
    
    plt.title(top_label,
              fontdict={'fontsize':12.},
              loc='right')
        
    # Second axis with typical precision
    ax2 = ax1.twinx()
    ax2.set_yscale('log')
              
    epsilon = 1.e-9
              
    def precision_to_information(precision):
        return (np.maximum(precision,epsilon) / 0.246) ** 4.

    def information_to_precision(fisher_inf):
        return 0.246 * np.maximum(fisher_inf,epsilon) ** 0.25
    
    def tick_function_information(fisher_inf):
        precision = 0.246 * np.maximum(fisher_inf,epsilon) ** 0.25
        return [str(z) for z in precision]
    
    def tick_function_precision(precision):
        return [str(z) for z in precision]
    
    ax2_limits = ax1.get_ylim()
    precision_limits = information_to_precision(ax2_limits)
    
    precision_ticks = np.array([0.1,0.2,0.5,1.,1.5,2.0,3.0,4.0,5.0])
    ax2_ticks = precision_to_information(precision_ticks)
    
    precision_minor_ticks = np.linspace(0.,5.,51)
    ax2_minor_ticks = precision_to_information(precision_minor_ticks)
    ax2.set_yticks(ax2_ticks,minor=False)
    ax2.set_yticklabels(tick_function_precision(precision_ticks))
    ax2.set_yticks(ax2_minor_ticks,minor=True)
    ax2.set_ylim(ax2_limits)
    
    ax2.set_ylabel(r'Reach $\Lambda / \sqrt{f}$ [TeV]')
    ax2.yaxis.set_label_coords(1.058,0.5)
    
    # legend
    if eigenvalue_operator_legend:
        
        if size_upper == 2:
            legend_labels = [r'Eigenvector composition:'] + operatorlabels
            legend_labels_x = [0.58,0.88,0.94]
            legend_labels_color = ['black'] + eigenvalue_colors
        else:
            legend_labels = [r'Eigenvector composition:'] + operatorlabels
            legend_labels_x = [0.94-(len(operatorlabels)-1)*0.1-0.4] + np.linspace(0.94-(len(operatorlabels)-1)*0.1, 0.94, num=2)
            legend_labels_color = ['black'] + eigenvalue_colors
        
        for legend_label,x,col in zip(legend_labels,legend_labels_x,legend_labels_color):
            ax1.text(x * base_xmax,legend_position,
                     legend_label,
                     fontsize=12,
                     color=col,
                     horizontalalignment='left',
                     verticalalignment='center')

    #################################################################################
    # Lower plot
    #################################################################################

    # Plot bars!
    ax3 = plt.subplot(212)

    bar_plot = ax3.bar(xpos_lower,
                   determinants,
                   width=width_lower,
                   log=False)

    for i in range(n_entries):
        bar_plot[i].set_color(bar_colors_light[categories[i]])
        bar_plot[i].set_edgecolor(bar_colors[categories[i]])
    
    ax3.set_xlim([base_xmin - 0.2,base_xmax + 0.2])
    ax3.set_ylim([0.,max(determinants)*1.05])
    
    ax3.set_xticks(xpos_ticks)
    ax3.set_xticklabels(labels, rotation=40, ha='right')
    if normalise_determinants:
        ax3.set_ylabel(r'$(\det \ I_{ij} / \det \ I_{ij}^{\mathrm{full}})^{1/' + str(size_lower) + r'}$')
    else:
        ax3.set_ylabel(r'$(\det \ I_{ij})^{1/' + str(size_lower) + r'}$')
    ax3.yaxis.set_label_coords(-0.052,0.5)

    if len(additional_label) > 0:
        ax3.text(0.99 * base_xmax,max(determinants) * 0.93,
                 additional_label,
                 fontsize=12,
                 color='black',
                 horizontalalignment='right',
                 verticalalignment='center')
        
    def precision_to_norm_information(precision):
        return (np.maximum(precision,epsilon) / 0.246) ** 4. / max_information

    def norm_information_to_precision(fisher_inf):
        return 0.246 * np.maximum(fisher_inf * max_information,epsilon) ** 0.25
    
    def tick_function_precision(precision):
        return [str(z) for z in precision]
    
    # Second axis with typical precision
    ax4 = ax3.twinx()
    ax4_limits = ax3.get_ylim()
    precision_limits = norm_information_to_precision(np.array(ax4_limits))
    
    #precision_ticks = np.array([1.e-3,1.e-2,1.e-1,1.,1.e1,1.e2,1.e3])
    precision_ticks = np.array([0.1,0.2,0.3,0.4,0.5,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5])
    ax4_ticks = precision_to_norm_information(precision_ticks)
    
    ## precision_minor_ticks = []
    ## for i in [0.01,0.1,1.]:
    ##     for j in range(10):
    ##         precision_minor_ticks.append(i * j)
    ## precision_minor_ticks.append(precision_ticks[-1])
    ## precision_minor_ticks = np.array(precision_minor_ticks)
    precision_minor_ticks = np.linspace(0.1,5.,50)
    ax4_minor_ticks = precision_to_norm_information(precision_minor_ticks)
    
    ax4.set_yticks(ax4_ticks,minor=False)
    ax4.set_yticklabels(tick_function_precision(precision_ticks))
    ax4.set_yticks(ax4_minor_ticks,minor=True)
    
    ax4.set_ylim(ax4_limits)
    
    ax4.set_ylabel(r'Reach $\Lambda / \sqrt{f}$ [TeV]')
    ax4.yaxis.set_label_coords(1.058,0.5)
    
    #################################################################################
    # Show and Save
    #################################################################################
    
    plt.show()

    def create_missing_folders(folders):
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
            elif not os.path.isdir(folder):
                raise OSError('Path {} exists, but is no directory!'.format(folder))

    create_missing_folders([os.path.dirname(filename)])
    fig.savefig(filename, dpi=300)
    plt.close()


def plot_fisherinfo_contours_2d(matrices_2d,
                                matrix_labels,
                                filename,
                                contour_distance=1.,
                                axes_max=1.,
                                xlabel='',
                                ylabel='',
                                n_points=100):
    
    """
    :matrices_2d: list of 2 x 2 fisher information matrices
    :matrix_labels:  list of labels corresponding to the fisher information matrices (list of strings)
    :filename: output filename (string)
    :contour_distance: distances drawn  (integer)
    :axes_max: maximum value on both axis  (integer)
    :xlabel: label of x-axis (string)
    :ylabel: label of y-axis (string)
    :n_points=100):
    """
    
    #################
    # general
    global global_A
    epsilon = 1.e-9
    
    n_matrices = len(matrices_2d)
    
    #function to evaluate scalar product
    def xAx(x,y):
        global global_A
        xvec = np.array([x,y])
        return xvec.dot(global_A.dot(xvec))
    
    vec_xAx = np.vectorize(xAx)
    
    # line styles
    matrix_color = ['black', 'red', 'blue', 'green', 'darkorange', 'fuchsia', 'turquoise', 'grey']*7*5
    matrix_linestyle = ['solid', 'dashed', 'dotted', 'dashdot', 'solid', 'dotted', 'dashed']*8*5
    matrix_linewidth = [1.5,1.5,2.,1.5,1.5,2.,1.5]*8*5
    
    styleindex=[t for t in range(n_matrices)]
    
    if len(matrix_labels) == 0:
        matrix_labels = ['']*n_matrices
    
    for i in range(n_matrices):
        assert len(matrices_2d[i]) == 2 , "Fisher Information is not 2D"
    
    #################
    # calculate xy data for tangent-space contour
    xvalues = np.linspace(-axes_max,axes_max,n_points)
    yvalues = np.linspace(-axes_max,axes_max,n_points)
    xy_y, xy_x = np.meshgrid(xvalues,yvalues)
    xy_linearized_distance = [np.zeros((n_points,n_points)) for i in range(n_matrices)]
    styleindex_counter = 0
    for i in range(n_matrices):
        styleindex[i] = styleindex_counter
        styleindex_counter += 1
        global_A = matrices_2d[i]
        xy_linearized_distance[i] = vec_xAx(xy_x,xy_y)
    
    #################
    # xy plot
    fig = plt.figure(figsize=(4.5,4.5))
    fig.subplots_adjust(left=0.17,right=0.97,bottom=0.17,top=0.97)

    for i in range(n_matrices):
        contour_levels = np.array([contour_distance**2.])
        contour_labels = [{contour_levels[0]:ml} for ml in matrix_labels]
    
        cs = plt.contour(xy_x,xy_y,xy_linearized_distance[i],
                         contour_levels,
                         colors=matrix_color[styleindex[i]],
                         linestyles=matrix_linestyle[styleindex[i]],
                         linewidths=matrix_linewidth[styleindex[i]])
        if len(matrix_labels[i]) > 0:
            plt.clabel(cs, cs.levels, inline=True, fontsize=12, fmt=contour_labels[i])

    plt.axes().set_xlim([-axes_max - epsilon,axes_max + epsilon])
    plt.axes().set_ylim([-axes_max - epsilon,axes_max + epsilon])
    plt.axes().set_aspect('equal')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axes().yaxis.set_label_coords(-0.13,0.5)

    #################
    # Show and Save

    def create_missing_folders(folders):
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
            elif not os.path.isdir(folder):
                raise OSError('Path {} exists, but is no directory!'.format(folder))

    plt.show()
    create_missing_folders([os.path.dirname(filename)])
    fig.savefig(filename, dpi=300)
    plt.close()

def kinematic_distribution_of_information(xbins,
                  xlabel,xmin,xmax,
                  xsecs,
                  matrices,
                  matrices_aux,
                  filename,
                  ylabel_addition='',
                  log_xsec=False,
                  norm_xsec=True,
                  show_aux=False,
                  show_labels=False,
                  label_pos_information=(0.,0.),
                  label_pos_sm=(0.,0.),
                  label_pos_bsm=(0.,0.),
                  label_pos_bkg=(0.,0.),
                  label_bsm=r''):
    
    epsilon = 1.e-9
    
    # Calculate data
    size = len(matrices[1])
    exponent = 1./float(size)
    determinants = [np.linalg.det(m)**exponent for m in matrices]
    determinants_aux = [np.linalg.det(m)**exponent for m in matrices_aux]
    
    determinants = np.nan_to_num(determinants)
    determinants_aux = np.nan_to_num(determinants_aux)
    
    # extract normalized xsec information
    if norm_xsec:
        norm = 1./max(sum([xs for xs in xsecs]), epsilon)
    else:
        norm = 1.
    xsec_norm = [norm * xs for xs in xsecs]

    n_entries = len(determinants)

    #Get xvals from xbins
    xvals = [(xbins[i]+xbins[i+1])/2 for i in range(0,len(xbins)-1)]
    assert len(xvals) == n_entries
    
    # Plotting options
    xs_color = 'black'
    xs_linestyle = 'solid'
    xs_linewidth = 1.5
    
    det_color = 'red'
    det_linestyle = 'solid'
    det_linewidth = 1.5
    det_alpha = 0.04
    
    det_aux_color = 'red'
    det_aux_linestyle = 'dashed'
    det_aux_linewidth = 1.5
    
    #################################################################################
    # Full plot
    #################################################################################
    
    fig = plt.figure(figsize=(5.4,4.5))
    ax1 = plt.subplot(111)
    fig.subplots_adjust(left=0.1667,right=0.8333,
                        bottom=0.17,top=0.97)
        
    if log_xsec:
        ax1.set_yscale('log')

    # SM signal
    ax1.hist(xvals,
         weights=xsec_norm,
         bins=xbins,
         range=(xmin,xmax),
         histtype='step',
         color=xs_color,
         linewidth=xs_linewidth,
         linestyle=xs_linestyle)


    # axis
    if norm_xsec:
        ax1.set_ylabel(r"Normalized distribution",
                   color=xs_color)
    else:
        ax1.set_ylabel(r"$\sigma$ [pb/bin]")
    ax1.set_xlim([xmin,xmax])
    ax1.set_ylim([0.,max(xsec_norm)*1.05])
    ax1.set_xlabel(xlabel)
    for tl in ax1.get_yticklabels():
        tl.set_color(xs_color)
    
    # plot: determinant
    ax2 = ax1.twinx()
    
    if show_aux:
        ax2.hist(xvals,
                 weights=determinants_aux,
                 bins=xbins,
                 range=(xmin,xmax),
                 histtype='step',
                 color=det_aux_color,
                 linewidth=det_aux_linewidth,
                 linestyle=det_aux_linestyle)

    ax2.hist(xvals,
             weights=determinants,
             bins=xbins,
             range=(xmin,xmax),
             histtype='stepfilled',
             alpha=det_alpha,
             color=det_color,
             linewidth=0.)

    ax2.hist(xvals,
         weights=determinants,
         bins=xbins,
         range=(xmin,xmax),
         histtype='step',
         color=det_color,
         linewidth=det_linewidth,
         linestyle=det_linestyle)
        
    ax2.set_xlim([xmin,xmax])
    ax2.set_ylim([0.,max(determinants)*1.1])
    ax2.set_ylabel(r"$(\det \; I_{ij})^{1/" + str(size) + "}$" + ylabel_addition,color=det_color)
    for tl in ax2.get_yticklabels():
        tl.set_color(det_color)


    #################################################################################
    # Show and Save
    #################################################################################

    plt.show()

    def create_missing_folders(folders):
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
            elif not os.path.isdir(folder):
                raise OSError('Path {} exists, but is no directory!'.format(folder))

    create_missing_folders([os.path.dirname(filename)])
    fig.savefig(filename, dpi=300)
    plt.close()


