import numpy as np

from madminer.utils import morphing as m




def original_find_components(parameter_max_power = [2,2]):
    # Test find_components with max_power = [2,2], without max_overall power
    morpher = m.PhysicsMorpher(parameter_max_power=parameter_max_power)
    components = morpher.find_components()
    expected_components = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
    assert components.any() == expected_components.any(), "The output of original_find_components is different from expected"

    # Test with the restriction of max_overall_power == 1
    components_max = morpher.find_components(1)
    expected_max = np.array([[0,0], [0,1], [1,0]])
    assert components_max.any() == expected_max.any(), "The max_power of outputs of original_find_components are not restricted to one"


def original_set_basis(basis_numpy = np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]]), 
                        parameter_max_power = [2,2], 
                        this_components = np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]) ):

    # test set basis with input of basis_numpy.
    morpher = m.PhysicsMorpher(parameter_max_power=parameter_max_power)
    morpher.set_components(this_components)
    morpher.set_basis(basis_numpy = basis_numpy)
    
    # Check if the basis is the same
    assert morpher.basis.any() == basis_numpy.any()

    expected_matrix = np.array([[1, 2.08333, 1.45833, 0.416667, 0.0416667], 
                            [-5, -10.1667, -6.83333, -1.83333, -0.166667], 
                            [10, 19.5, 12.25, 3, 0.25], 
                            [-10, -17.8333, -9.83333, -2.16667, -0.166667], 
                            [5, 6.41667, 2.95833, 0.583333, 0.0416667]]).T

    # the set_basis also calls calculate matrix if self.morphing_matrix is not defined, check if the morphing matrix is expected 
    assert morpher.morphing_matrix.any() == expected_matrix.any(), "The calculated matrix in set_basis is different from expected."

def original_calculate_matrix(this_basis = np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]]), 
                                this_components = np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]),):

    morpher = m.PhysicsMorpher(parameter_max_power=[2,2])

    # Set the powers of components and basis
    morpher.set_components(this_components)
    morpher.set_basis(basis_numpy= this_basis)

    # Calculate the morphing matrix
    morphing_matrix = morpher.calculate_morphing_matrix()
    
    # The expected output values of the matrix based on the inputes. 
    expected_matrix = np.array([[1, 2.08333, 1.45833, 0.416667, 0.0416667], 
                                [-5, -10.1667, -6.83333, -1.83333, -0.166667], 
                                [10, 19.5, 12.25, 3, 0.25], 
                                [-10, -17.8333, -9.83333, -2.16667, -0.166667], 
                                [5, 6.41667, 2.95833, 0.583333, 0.0416667]]).T
    
    # assert the matrix is the same as expected
    assert morphing_matrix.all() == expected_matrix.all(), "The output matrix is different from expected"

    output_weights = morpher.calculate_morphing_weights(theta=[1,1])
    # Test the if the output weights are as expected
    expected_weights = np.array([5, -24, 45, -40, 15])
    assert output_weights.all() == expected_weights.all(), "The output weights are not as expected"


def original_weight_gradients(this_components = np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]),
                    this_basis = np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]])):

    # Morpher object with the initial format of inputs
    old_morpher = m.PhysicsMorpher(parameter_max_power=[2,2])
    old_morpher.set_components(this_components)
    old_morpher.set_basis(basis_numpy= this_basis)

    # calculate and check the output values of the old and new mophing weights. 
    old_weight_gradient = old_morpher.calculate_morphing_weight_gradient(theta = [1,1])

    expected_weight_gradient = np.array([[13.58333333, -66, 126, -115.33333333, 45.75],
                                         [6.41666667, -30, 54, -44.66666667, 14.25]])

    assert expected_weight_gradient.any() == old_weight_gradient.any(), "The weight gradient differs from expected value"



def test_morphing():
    parameter_max_power = [2,2]
    this_components = np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]) 
    this_basis = np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]])

    # This test only tests with the selected values of parameter_max_power = [2,2]
    original_find_components(parameter_max_power=parameter_max_power)

    # Test how the set_basis works from the original code. 
    original_set_basis(parameter_max_power= parameter_max_power, this_components= this_components, basis_numpy = this_basis)

    # Test if the output matrix and weights are as expected.
    # The expected output is calculated with this sample only. 
    original_calculate_matrix(this_basis=this_basis, this_components=this_components)

    # Test if weight_gradient is as expected. 
    original_weight_gradients()
    




