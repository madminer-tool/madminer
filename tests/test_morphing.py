import numpy as np

from madminer.utils import morphing as m


def test_original_find_components(parameter_max_power=[2, 2]):
    # Test find_components with max_power = [2,2], without max_overall power
    morpher = m.PhysicsMorpher(parameter_max_power=parameter_max_power)
    components = morpher.find_components()
    expected_components = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    )
    assert np.allclose(
        components, expected_components
    ), "The output of original_find_components is different from expected"

    # Test with the restriction of max_overall_power == 1
    components_max = morpher.find_components(1)
    expected_max = np.array([[0, 0], [0, 1], [1, 0]])
    assert np.allclose(
        components_max, expected_max
    ), "The max_power of outputs of original_find_components are not restricted to one"


def test_original_set_basis(
    basis_numpy=np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]]),
    parameter_max_power=[2, 2],
    this_components=np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]),
):
    # test set basis with input of basis_numpy.
    morpher = m.PhysicsMorpher(parameter_max_power=parameter_max_power)
    morpher.set_components(this_components)
    morpher.set_basis(basis_numpy=basis_numpy)

    # Check if the basis is the same
    assert np.allclose(
        morpher.basis, basis_numpy, rtol=10**-5
    ), "The set_basis does not set basis correctly"

    expected_matrix = np.array(
        [
            [1, 2.08333, 1.45833, 0.416667, 0.0416667],
            [-5, -10.1667, -6.83333, -1.83333, -0.166667],
            [10, 19.5, 12.25, 3, 0.25],
            [-10, -17.8333, -9.83333, -2.16667, -0.166667],
            [5, 6.41667, 2.95833, 0.583333, 0.0416667],
        ]
    ).T
    # the set_basis also calls calculate matrix if self.morphing_matrix is not defined, check if the morphing matrix is expected
    assert np.allclose(
        morpher.morphing_matrix, expected_matrix, rtol=10**-5
    ), "The calculated matrix in set_basis is different from expected."


def test_original_calculate_matrix(
    this_basis=np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]]),
    this_components=np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]),
):
    morpher = m.PhysicsMorpher(parameter_max_power=[2, 2])

    # Set the powers of components and basis
    morpher.set_components(this_components)
    morpher.set_basis(basis_numpy=this_basis)

    # Calculate the morphing matrix
    morphing_matrix = morpher.calculate_morphing_matrix()

    # The expected output values of the matrix based on the inputes.
    expected_matrix = np.array(
        [
            [1, 2.08333, 1.45833, 0.416667, 0.0416667],
            [-5, -10.1667, -6.83333, -1.83333, -0.166667],
            [10, 19.5, 12.25, 3, 0.25],
            [-10, -17.8333, -9.83333, -2.16667, -0.166667],
            [5, 6.41667, 2.95833, 0.583333, 0.0416667],
        ]
    ).T

    assert np.allclose(
        morphing_matrix, expected_matrix, rtol=10**-5
    ), "The output matrix is different from expected"
    output_weights = morpher.calculate_morphing_weights(theta=[1, 1])

    # Test the if the output weights are as expected
    expected_weights = np.array([5, -24, 45, -40, 15])
    assert np.allclose(output_weights, expected_weights), "The output weights are not as expected"


def test_original_weight_gradients(
    this_components=np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]),
    this_basis=np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]]),
):
    # Morpher object with the initial format of inputs
    old_morpher = m.PhysicsMorpher(parameter_max_power=[2, 2])
    old_morpher.set_components(this_components)
    old_morpher.set_basis(basis_numpy=this_basis)

    # calculate and check the output values of the old and new mophing weights.
    old_weight_gradient = old_morpher.calculate_morphing_weight_gradient(theta=[1, 1])

    expected_weight_gradient = np.array(
        [[13.58333333, -66, 126, -115.33333333, 45.75], [6.41666667, -30, 54, -44.66666667, 14.25]]
    )

    assert np.allclose(
        old_weight_gradient, expected_weight_gradient, rtol=10**-5
    ), "The weight gradient differs from expected value"


def test_compare_weights(
    gs=np.array([[1, 1, 1, 1, 1], [-5, -4, -3, -2, -1]]),
    gp=None,
    gd=None,
    parameter_max_power=[2, 2],
    this_components=np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]),
    this_basis=np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]]),
    theta=np.array([1, 1]),
):
    # Test if the old and new inputs will output the same matrix with the same inputs.
    old_morpher = m.PhysicsMorpher(
        parameter_max_power=parameter_max_power,
    )
    old_morpher.set_components(this_components)
    old_morpher.set_basis(basis_numpy=this_basis)

    new_morpher = m.PhysicsMorpher(
        parameter_max_power=parameter_max_power,
    )
    _ = new_morpher.find_components(Ns=2, Nd=0, Np=0)
    new_morpher.set_basis(basis_d=gd, basis_p=gp, basis_s=gs)

    # Both morphing matrix was calculated in set_basis procedure.
    # Test if the output matrix is the same for both form of inputs
    assert np.allclose(
        new_morpher.morphing_matrix, old_morpher.morphing_matrix
    ), " The morphing matrices calculated by the new and old method are different"

    # Test if the weights for the given theta equals
    assert np.allclose(
        new_morpher.calculate_morphing_weights(theta), old_morpher.calculate_morphing_weights(theta)
    ), " The output weights of two methods are different"


def test_overdetermine(
    gd=None, gp=None, gs=np.array([[1, 1, 1, 1, 1, 1, 1], [-5, -4, -3, -2, -1, 0, 1]])
):
    # Check if the improved method can do overdetermined morphing as expected with the new input format
    new_morpher = m.PhysicsMorpher(parameter_max_power=[4, 4])
    _ = new_morpher.find_components(Ns=2, Nd=0, Np=0)
    new_morpher.set_basis(basis_d=gd, basis_p=gp, basis_s=gs)

    expected = np.array(
        [
            [-0.04112554, -0.07936508, 0.05681818, 0.06313131, 0.01136364],
            [0.15151515, 0.23412698, -0.21590909, -0.18434343, -0.02651515],
            [-0.14069264, -0.06349206, 0.18560606, 0.05808081, 0.00378788],
            [-0.12987013, -0.33333333, 0.28030303, 0.18181818, 0.02272727],
            [0.33549784, -0.26984127, -0.14772727, 0.00252525, 0.00378788],
            [0.77056277, 0.09920635, -0.54924242, -0.23989899, -0.02651515],
            [0.05411255, 0.41269841, 0.39015152, 0.11868687, 0.01136364],
        ]
    ).T
    assert np.allclose(
        new_morpher.calculate_morphing_matrix(), expected
    ), "The overdetermined matrix is different from expected"

def test_new_find_components(Nd=0, Np=0, Ns=2):
    # Test find components with respecting Nd, Np, Ns values
    new_morpher = m.PhysicsMorpher(parameter_max_power=[4, 4])
    assert np.allclose(
        new_morpher.find_components(Ns=Ns, Nd=Nd, Np=Np),
        np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]),
    ), " The output differs from expected components"

    Nd = 1
    Np = 1
    Ns = 0
    assert np.allclose(
        new_morpher.find_components(Ns=Ns, Nd=Nd, Np=Np), np.array([2, 2])
    ), " The output differs from expected components"

    Nd = 0
    Np = 0
    Ns = 1
    assert np.allclose(
        new_morpher.find_components(Ns=Ns, Nd=Nd, Np=Np), np.array([4])
    ), " The output differs from expected components"

    Nd = 2
    Np = 1
    Ns = 0
    assert np.allclose(
        new_morpher.find_components(Ns=Ns, Nd=Nd, Np=Np),
        np.array([[2, 0, 2], [1, 1, 2], [0, 2, 2]]),
    ), " The output differs from expected components"


def test_compare_weight_gradients(
    gs=np.array([[1, 1, 1, 1, 1], [-5, -4, -3, -2, -1]]),
    gp=None,
    gd=None,
    parameter_max_power=[2, 2],
    this_components=np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]),
    this_basis=np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]]),
):

    # Test if the weight gradients output the same with the new input format
    # Morpher object with the new format of inputs
    new_morpher = m.PhysicsMorpher(
        parameter_max_power=parameter_max_power,
    )

    _ = new_morpher.find_components(Ns=2, Nd=0, Np=0)
    new_morpher.set_basis(basis_d=gd, basis_p=gp, basis_s=gs)

    # Morpher object with the initial format of inputs
    old_morpher = m.PhysicsMorpher(parameter_max_power=[2, 2])
    old_morpher.set_components(this_components)
    old_morpher.set_basis(basis_numpy=this_basis)

    # calculate and check the output values of the old and new mophing weights.
    old_weight_gradient = old_morpher.calculate_morphing_weight_gradient(theta=[1, 1])
    new_weight_gradient = new_morpher.calculate_morphing_weight_gradient(theta=[1, 1])

    assert np.allclose(
        new_weight_gradient, old_weight_gradient
    ), "The weight gradient of old inputs and the new inputs are different."


def test_evaluate_morphing():
    # Check if the morpher object created with old and new inputs can execute evaluate_morphing() function
    old_morpher = m.PhysicsMorpher(
        parameter_max_power=[4, 4],
        parameter_range=[
            (
                -1.0,
                1.0,
            ),
            (-1.0, 1.0),
        ],
    )
    this_components = np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]])
    this_basis = np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]])

    # Set the powers of components and basis
    old_morpher.set_components(this_components)
    old_morpher.set_basis(basis_numpy=this_basis)

    old_value = old_morpher.evaluate_morphing(basis=this_basis)

    # Morpher object with the new format of inputs
    new_morpher = m.PhysicsMorpher(
        parameter_max_power=[4, 4],
        parameter_range=[
            (
                -1.0,
                1.0,
            ),
            (-1.0, 1.0),
        ],
    )

    gd = None
    gp = None
    gs = np.array([[1, 1, 1, 1, 1], [-5, -4, -3, -2, -1]])

    _ = new_morpher.find_components(Ns=2, Nd=0, Np=0)
    new_morpher.set_basis(basis_d=gd, basis_p=gp, basis_s=gs)

    new_value = new_morpher.evaluate_morphing(basis=this_basis)

    assert np.allclose(
        new_value, old_value, rtol=10 ^ 3
    ), "The output values of old and new evaluate_morphing with same inputs varies more than 10^3"


def test_new_weights():
    # In the order of gd, gp, gc, the code will determine the number of each coupling parameter based on gd, gp, gc...
    n_d = 1
    n_p = 3
    n_s = 0

    # specify gd, gp, gc separately
    gd = np.array([[1, 1, 1, 1, 1, 1]])
    gp = np.array(
        [
            [0.7071, 0.7071, 0.7071, 0.7071, 0.7071, 0.7071],
            [0, 4.2426, 0, 4.2426, -4.2426, 0],
            [0, 0, 4.2426, 4.2426, 0, -4.2426],
        ]
    )
    gs = None  # np.array([[1,1,1,1,1], [-5, -4, -3, -2, -1]])

    xsec = np.array(
        [0.515, 0.732, 0.527, 0.742, 0.354, 0.527, 0.364, 0.742, 0.364, 0.621, 0.432, 0.621, 0.432]
    )  # define once, the code will take the corresponding xsec values for the morphing weights

    new_morpher = m.PhysicsMorpher(parameter_max_power=[4, 4])

    new_morpher.find_components(Nd=n_d, Np=n_p, Ns=n_s)

    new_morpher.set_basis(basis_d=gd, basis_p=gp, basis_s=gs)

    weights = new_morpher.calculate_morphing_weights(theta_d=[1], theta_p=[0.7071, 0, 0])

    predict_xsec = _calculate_predict_xsec(xsec=xsec, morphing_weights=weights)

    assert np.allclose(
        predict_xsec, xsec[0]
    ), "The predict xsec value does not match the simulated value."


def test_find_components_BSM():
    # The BSM_max_power will limit the max_power other than gd_0, gp_0, gs_0
    # Will eliminate the component group if the exceed the BSM_max_power

    Nd = 2
    Np = 1
    Ns = 0

    new_morpher = m.PhysicsMorpher(parameter_max_power=[4, 4])

    # If no BSM_max_power is set, the output should be np.array([[2, 0, 2], [1, 1, 2], [0, 2, 2]]
    # in this case, components[0] = gd_0, components[1] = gd_1, components_2 = gp_0
    # Thus this max power should only keep the values gd_1 <= 0 (The only BSM coupling), which should be only [2,0,2]
    components = new_morpher.find_components(Ns=Ns, Nd=Nd, Np=Np, BSM_max_power=0)

    expected_components = np.array([[2, 0, 2]])
    assert np.allclose(
        components, expected_components
    ), " The output differs from expected components with BSM_max"

    Nd = 2
    Np = 0
    Ns = 2

    new_morpher = m.PhysicsMorpher(parameter_max_power=[4, 4])

    # in this case, components[0] = gd_0, components[1] = gd_1, components[2] = gs_0, components[3] = gs_1
    # Thus this max power should only keep the values gd_1 and gs_1 <= 0 (The only BSM coupling), which should be only [2,0,2]
    components = new_morpher.find_components(Ns=Ns, Nd=Nd, Np=Np, BSM_max_power=0)

    expected_components = np.array([[2, 0, 2, 0], [1, 0, 3, 0], [0, 0, 4, 0]])

    np.allclose(
        components, expected_components
    ), "The output differs from expected components with BSM_max"


def test_get_min_basis():
    new_morpher = m.PhysicsMorpher(parameter_max_power=[4, 4])

    expected_value = 3
    calculated_value = new_morpher.get_min_basis(Nd=2, Np=1)

    np.allclose(
        calculated_value, expected_value
    ), "The generated minimum number of basis requires differs from expected"

    expected_value = 5
    calculated_value = new_morpher.get_min_basis(Ns=2)

    np.allclose(
        calculated_value, expected_value
    ), "The generated minimum number of basis requires differs from expected"
    return


# helper method that calculate W_i and Neff/xsec with W_i = w_i*sigma_i and Neff = sum(W_i)
def _calculate_predict_xsec(xsec, morphing_weights):
    index = len(morphing_weights)
    if len(xsec) < index:
        raise Exception("The number of xsec values is smaller than the number of morphing weights")

    # Get the corresponding xsec values for the morphing weights
    this_xsec = xsec[:index]
    W_i = np.multiply(this_xsec, morphing_weights, dtype=np.float32)
    return sum(W_i)
