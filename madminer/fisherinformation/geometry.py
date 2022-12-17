import logging
import random

import numpy as np

from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.stats import chi2

from ..utils.various import load_and_check

logger = logging.getLogger(__name__)


class InformationGeometry:
    """
    Functions to calculate limits using Information Geometry.

    After initializing the `InformationGeometry` class, a Fisher Information needs to be provided using
    one of the following functions

    * `InformationGeometry.information_from_formula()` defines the Fisher Information
      explicitly as function of the theory parameters `theta`.
    * `InformationGeometry.information_from_grid()` loads a grid of Fisher Information
      which is then interpolated.

    Using information geometrical methods, the function `InformationGeometry.distance_contours()` then
    calculates the distance contours and equivalently the p-values throughout parameter space.
    """

    def __init__(self):
        self.infotype = None
        self.dimension = 0
        self.information_formula = None
        self.inverse = "exact"

    def information_from_formula(self, formula, dimension):
        """
        Explicitly defines the Fisher Information as function of the theory parameter `theta`
        through a formula that can be evaluated using `eval()`.

        Parameters
        ----------
        formula : str
            Explicit definition of the Fisher Information as ndarray, which can be a function of
            the n-dimensional theory parameter `theta`.
            Example: formula="np.array([[1+theta[0],1],[1,2*theta[1]**2]])"

        dimension : int
            Dimensionality of the theory parameter space.
        """

        self.infotype = "formula"
        self.dimension = dimension
        self.information_formula = formula

    def information_from_grid(self, theta_grid, fisherinformation_grid, option="smooth", inverse="exact"):
        """
        Loads a grid of coordinates and corresponding Fisher Information, which is then interpolated.

        Parameters
        ----------
        theta_grid : ndarray
            List if parameter points `theta` at which the Fisher information matrices `I_ij(theta)`
            is evaluated. Shape (n_gridpoints, n_dimension).

        fisherinformation_grid : ndarray
            List if Fisher information matrices `I_ij(theta)`. Shape (n_gridpoints, n_dimension, n_dimension).

        option : {"smooth", "linear"}
            Defines if the Fisher Information is interpolated smoothly using the function
            CloughTocher2DInterpolator() or piecewise linear using LinearNDInterpolator().
            Default = 'smooth'.

        inverse : {"exact", "interpolate"}
            Defines if the inverse Fisher Information is obtained by either first interpolating
            the Fisher Information and then inverting it ("exact") or by first inverting the grid
            of Fisher Informations and then interpolating the inverse ("interpolate"). Default = 'exact'.
        """

        self.infotype = "grid"
        self.inverse = inverse

        # load from file
        theta_grid = load_and_check(theta_grid)
        fisherinformation_grid = load_and_check(fisherinformation_grid)
        self.dimension = len(fisherinformation_grid[0])

        # Interpolate Information
        if option == "linear":
            self.infofunction = LinearNDInterpolator(points=theta_grid, values=np.array(fisherinformation_grid))
        elif option == "smooth":
            self.infofunction = CloughTocher2DInterpolator(points=theta_grid, values=np.array(fisherinformation_grid))
        else:
            RuntimeError(f"Option unknown: {option}")

        # Interpolate inverse information
        if self.inverse == "interpolate":
            inv_fisherinformation_grid = np.array([np.linalg.inv(info) for info in fisherinformation_grid])
            if option == "linear":
                self.infofunction_inv = LinearNDInterpolator(points=theta_grid, values=inv_fisherinformation_grid)
            elif option == "smooth":
                self.infofunction_inv = CloughTocher2DInterpolator(points=theta_grid, values=inv_fisherinformation_grid)

    def _information(self, theta):
        """
        Low level function that calculates the Fisher Information as function of
        the theory parameter `theta`

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the Fisher information matrix `I_ij(theta)` is evaluated.

        Returns
        -------
        fisher_information : ndarray
            Fisher information matrix with shape `(n_dimension, n_dimension)`.
        """

        # check input format
        assert len(theta) == self.dimension, "theta should have length %r, not %r" % (self.dimension, len(theta))

        # calculate information
        if self.infotype == "formula":
            information = eval(self.information_formula)
        elif self.infotype == "grid":
            information = self.infofunction(tuple(theta))
        else:
            raise RuntimeError("Information not defined yet")

        # check output format
        assert np.shape(information) == (self.dimension, self.dimension), "information should have shape %r, not %r" % (
            (self.dimension, self.dimension),
            np.shape(information),
        )

        return information

    def _information_inv(self, theta):
        """
        Low level function that calculates the inverse Fisher Information as function of
        the theory parameter `theta`.

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the inverse Fisher information
            matrix `I^{-1}_ij(theta)` is evaluated.

        Returns
        -------
        inverse_fisher_information : ndarray
            Inverse Fisher information matrix with shape `(n_dimension, n_dimension)`.
        """

        if self.inverse == "interpolate":
            return self.infofunction_inv(tuple(theta))
        else:
            return np.linalg.inv(self._information(theta))

    def _information_derivative(self, theta):
        """
        Low level function that calculates the derivative of Fisher Information
        `\partial_k I_{ij}` at the theory parameter `theta`.

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the derivative of the Fisher information
            matrix is evaluated.

        Returns
        -------
        fisher_information_derivative : ndarray
            Derivative of Fisher information matrix with shape `(n_dimension, n_dimension, n_dimension)`.
        """

        epsilon = 10**-3
        dtheta = np.identity(len(theta)) * epsilon
        return np.array(
            [(self._information(theta + dtheta[k]) - self._information(theta)) / epsilon for k in range(len(theta))]
        )

    def _christoffel(self, theta):
        """
        Low level function that calculates the Christoffel symbol (2nd kind) Gamma^i_jk at
        the theory parameter `theta`.  Here Gamma^i_jk=0.5*I^{im}(\partial_k I_{mj}
        + \partial_j I_{mk} - \partial_m I_{jk})

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the Christoffel symbol is evaluated.

        Returns
        -------
        christoffel_symbol : ndarray
            Christoffel symbol with shape `(n_dimension, n_dimension, n_dimension)`.
        """

        term1 = np.einsum("ad,cdb->abc", self._information_inv(theta), self._information_derivative(theta))
        term2 = np.einsum("ad,bdc->abc", self._information_inv(theta), self._information_derivative(theta))
        term3 = np.einsum("ad,bcd->abc", self._information_inv(theta), self._information_derivative(theta))
        return 0.5 * (term1 + term2 - term3)

    def find_trajectory(self, theta0, dtheta0, limits, stepsize=1):
        """
        Finds the geodesic trajectory starting at a parameter point theta0 going in the
        initial direction dtheta0.

        Parameters
        ----------
        theta0 : ndarray
            Parameter point `theta0` at which the geodesic trajectory starts.

        dtheta0 : ndarray
            Initial direction `dtheta0` of the geodesic

        limits : list of (tuple of float)
            Specifies the boundaries of the parameter grid in which the trajectory
            is evaluated. It should be `[[min, max], [min, max], ..., [min, max]`,
            where the list goes over all parameters and `min` and `max` are float.

        stepsize : int, optional
            Maximal stepsize `|Delta theta|` during numerical integration in parameter space.
            $Default: 1

        Returns
        -------
        list_of_theta : ndarray
            List of parameter points theta `(n_points, n_dimension)`.

        list_of_distance : ndarray
            List of distances from the staring point theta0 `(n_points, )`.

        """

        # initiate starting point
        theta = 1.0 * theta0
        dtheta = 1.0 * dtheta0
        dist = 0
        output_theta = [1.0 * theta]
        output_dist = [0]

        # calculate free-fall trajectory
        counter = 0
        in_grid = True
        while in_grid and counter < 200:
            counter += 1
            # normalize dtheta to stepsize
            dtheta = dtheta / np.linalg.norm(dtheta)

            # calculate ddtheta and distance
            ddtheta = -1.0 * np.einsum("abc,b,c->a", self._christoffel(theta), dtheta, dtheta)
            ddist = np.sqrt(np.einsum("ab,a,b", self._information(theta), dtheta, dtheta))

            # determine stepsize to be used
            max_stepsize = 0.05 * np.linalg.norm(dtheta) / np.linalg.norm(ddtheta)
            use_stepsize = min(max_stepsize, stepsize)

            # update theta,dtheta, dist
            theta += dtheta * use_stepsize
            dtheta += ddtheta * use_stepsize
            dist += ddist * use_stepsize

            # save
            theta = np.array(theta)
            if np.isnan(dist):
                break
            output_theta.append(theta * 1.0)
            output_dist.append(dist * 1.0)

            # check if outside range
            for th, lim in zip(theta, limits):
                if th < lim[0] or th > lim[1]:
                    in_grid = False

        return np.array(output_theta), output_dist

    def distance_contours(
        self,
        theta0,
        grid_ranges,
        grid_resolutions,
        stepsize=None,
        ntrajectories=None,
        continous_sampling=False,
        return_trajectories=False,
    ):
        """
        Finds the distance values from the point theta0 and the corresponding p-value
        within the parameter space bounded by `grid_ranges`.

        Parameters
        ----------
        theta0 : ndarray
            Parameter point `theta0` at which the geodesic trajectory starts.

        grid_ranges : list of (tuple of float)
            Specifies the boundaries of the parameter grid in which the trajectory
            is evaluated. It should be `[[min, max], [min, max], ..., [min, max]`,
            where the list goes over all parameters and `min` and `max` are float.

        grid_resolutions : list of int
            Resolution of the parameter space grid on which the p-values are evaluated.
            The individual entries specify the number of points along each parameter individually.

        stepsize : float or None, optional
            Maximal stepsize `|Delta theta|` during numerical integration in parameter space.
            If None, stepsize = min([(max-min)/20 for (min,max) in grid_ranges]). Default: None

        ntrajectories : int or None, optional
            Number of sampled trajectories. If None, ntrajectories = 20 times the
            number of dimensions. Default: None

        continous_sampling : bool, optional
            If n_dimension is 2, the trajectories are sampled continously in the angular
            direction. Default: False

        return_trajectories : bool, optional
            Returns the trajectories (parameter points and distances). Default: False

        Returns
        -------
        theta_grid : ndarray
            Parameter points at which the p-values are evaluated with shape `(n_grid_points, n_dimension)`.

        p_values : ndarray
            Observed p-values for each parameter point on the grid, with shape `(n_grid_points,)`.

        p_values : ndarray
            Interpolated distance from theta0 for each parameter point on the grid,
            with shape `(n_grid_points,)`.

        (list_of_theta, list_of_distance) : (ndarray,ndarray)
            Only returned if return_trajectories is True. List of parameter points
            theta `(n_points, n_dimension)` and List of distances from the
            staring point theta0 `(n_points, )`.

        """

        # automatic setting of stepsize and ntrajectories
        if stepsize is None:
            stepsize = min([(limit[1] - limit[0]) / 20.0 for limit in grid_ranges])
        if ntrajectories is None:
            ntrajectories = 20 * self.dimension
        if self.dimension is not 2:
            continous_sampling = False

        limits = (1.0 + 2.0 * stepsize) * np.array(grid_ranges)

        # determine trajectories
        thetas = []
        distances = []

        for i in range(ntrajectories):
            if continous_sampling:
                angle = 2.0 * np.pi / float(ntrajectories) * i
                dth0 = np.array([np.cos(angle), np.sin(angle)])
            else:
                dth0 = np.array([random.uniform(-1, 1) for _ in range(self.dimension)])

            logger.debug(f"Calculate Trajectory Number {i} with dtheta0={dth0}")
            ths, ds = self.find_trajectory(theta0, dth0, limits, stepsize)

            thetas.extend(ths)
            distances.extend(ds)

        thetas = np.array(thetas)

        # Create Theta Grid
        theta_grid_each = self._make_theta_grid_each(grid_ranges, grid_resolutions)
        theta_grid = self._make_theta_grid(grid_ranges, grid_resolutions)

        # Create Distance Grid
        distance_grid = griddata(thetas, distances, (theta_grid_each[0], theta_grid_each[1]), method="linear")

        # Create p-value Grid
        p_value_grid = self._asymptotic_p_value(distance_grid)

        # return
        if return_trajectories:
            return theta_grid, p_value_grid, distance_grid, (thetas, distances)
        else:
            return theta_grid, p_value_grid, distance_grid

    def _make_theta_grid_each(self, grid_ranges, grid_resolutions):
        theta_each = []
        for resolution, (theta_min, theta_max) in zip(grid_resolutions, grid_ranges):
            theta_each.append(np.linspace(theta_min, theta_max, resolution))
        theta_grid_each = np.meshgrid(*theta_each, indexing="ij")
        return theta_grid_each

    def _make_theta_grid(self, grid_ranges, grid_resolutions):
        theta_grid_each = self._make_theta_grid_each(grid_ranges, grid_resolutions)
        theta_grid_each = [theta.flatten() for theta in theta_grid_each]
        theta_grid = np.vstack(theta_grid_each).T
        return theta_grid

    def _asymptotic_p_value(self, dist):
        """
        Low level function to convert distances in p-values
        """
        p_value = chi2.sf(x=dist * dist, df=self.dimension)
        return p_value
