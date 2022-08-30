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
    # check if the morphing matrix is expected 
    assert morpher.morphing_matrix.any() == expected_matrix.any(), "The calculated matrix in set_basis is different from expected."

def original_calculate_matrix(this_basis = np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]]), 
                                this_components = np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]),):
    morpher = m.PhysicsMorpher(parameter_max_power=[2,2])

    # Set the powers of components and basis
    morpher.set_components(this_components)
    morpher.set_basis(basis_numpy= this_basis)

    morphing_matrix = morpher.calculate_morphing_matrix()
    
    # The expected output values of the matrix based on the inputes. 
    expected_matrix = np.array([[1, 2.08333, 1.45833, 0.416667, 0.0416667], 
                                [-5, -10.1667, -6.83333, -1.83333, -0.166667], 
                                [10, 19.5, 12.25, 3, 0.25], 
                                [-10, -17.8333, -9.83333, -2.16667, -0.166667], 
                                [5, 6.41667, 2.95833, 0.583333, 0.0416667]]).T
    
    assert morphing_matrix.all() == expected_matrix.all(), "The output matrix is different from expected"
    output_weights = morpher.calculate_morphing_weights(theta=[1,1])

    # Test the if the output weights are as expected
    expected_weights = np.array([5, -24, 45, -40, 15])
    assert output_weights.all() == expected_weights.all(), "The output weights are not as expected"


def test_original_mophing():
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


def compare_matrices_weights(gs = np.array([[1,1,1,1,1], [-5, -4, -3, -2, -1]]),
                    gp = None, gd = None, parameter_max_power = [2,2],
                    this_components = np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]),
                    this_basis = np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]]),
                    theta = np.array([1,1])):
    # Test if the old and new inputs will output the same matrix with the same inputs.  
    old_morpher = m.PhysicsMorpher(parameter_max_power=parameter_max_power, parameter_range=[(-1.,1.,), (-1.,1.)])
    old_morpher.set_components(this_components)
    old_morpher.set_basis(basis_numpy= this_basis)

    new_morpher = m.PhysicsMorpher(parameter_max_power=parameter_max_power, parameter_range=[(-1.,1.,), (-1.,1.)])
    _ = new_morpher.find_components(Ns = 2, Nd = 0, Np= 0)
    new_morpher.set_basis(basis_d = gd, basis_p = gp, basis_s=gs)

    # Both morphing matrix was calculated in set_basis procedure. 
    assert new_morpher.morphing_matrix.any() == old_morpher.morphing_matrix.any(), " The morphing matrices calculated by the new and old method are different"

    # Test if the weights for the given theta equals
    assert new_morpher.calculate_morphing_weights(theta).any() == old_morpher.calculate_morphing_weights(theta).any(), " The output weights of two methods are different"

def overdetermined(gd = None, gp = None, gs = np.array([[1,1,1,1,1], [-5, -4, -3, -2, -1]])):
    # Check if the improved method can do overdetermined morphing. 
    new_morpher = m.PhysicsMorpher(parameter_max_power=[4,4], parameter_range=[(-1.,1.,), (-1.,1.)])
    _ = new_morpher.find_components(Ns = 2, Nd = 0, Np= 0)
    new_morpher.set_basis(basis_d = gd, basis_p = gp, basis_s=gs)

    expected =  np.array([[-0.04112554, -0.07936508,  0.05681818,  0.06313131,  0.01136364],
                          [ 0.15151515, 0.23412698, -0.21590909, -0.18434343, -0.02651515],
                          [-0.14069264, -0.06349206,  0.18560606,  0.05808081,  0.00378788],
                          [-0.12987013, -0.33333333,  0.28030303,  0.18181818,  0.02272727],
                          [ 0.33549784, -0.26984127, -0.14772727,  0.00252525,  0.00378788],
                          [ 0.77056277, 0.09920635, -0.54924242, -0.23989899, -0.02651515],
                          [ 0.05411255, 0.41269841,  0.39015152,  0.11868687,  0.01136364]]).T
    assert new_morpher.calculate_morphing_matrix().any() == expected.any(), "The overdetermined matrix is different from expected"

def new_find_components(Nd=0, Np=0, Ns=2):
    # Test find components with respecting Nd, Np, Ns values
    new_morpher = m.PhysicsMorpher(parameter_max_power=[4,4], parameter_range=[(-1.,1.,), (-1.,1.)])
    assert new_morpher.find_components(Ns = Ns, Nd = Nd, Np= Np).any() == np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]).any(), " The output differs from expected components"

    Nd = 1
    Np = 1
    Ns = 0
    assert new_morpher.find_components(Ns = Ns, Nd = Nd, Np= Np).any() == np.array([[2,2]]).any(), " The output differs from expected components"   

    Nd = 0
    Np = 0
    Ns = 1
    assert new_morpher.find_components(Ns = Ns, Nd = Nd, Np= Np).any() == np.array([[2,2]]).any(), " The output differs from expected components"   

    Nd = 2
    Np = 1
    Ns = 0
    assert new_morpher.find_components(Ns = Ns, Nd = Nd, Np= Np).any()== np.array([[2,0,2], [1,1,2], [0,2,2]]).any(), " The output differs from expected components"   


def weight_gradients(gs = np.array([[1,1,1,1,1], [-5, -4, -3, -2, -1]]),
                    gp = None, gd = None, parameter_max_power = [2,2],
                    this_components = np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]),
                    this_basis = np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]])):

    # Morpher object with the new format of inputs
    new_morpher = m.PhysicsMorpher(parameter_max_power=parameter_max_power, parameter_range=[(-1.,1.,), (-1.,1.)])

    _ = new_morpher.find_components(Ns = 2, Nd = 0, Np= 0)
    new_morpher.set_basis(basis_d = gd, basis_p = gp, basis_s=gs)

    # Morpher object with the initial format of inputs
    old_morpher = m.PhysicsMorpher(parameter_max_power=[2,2])
    old_morpher.set_components(this_components)
    old_morpher.set_basis(basis_numpy= this_basis)

    # calculate and check the output values of the old and new mophing weights. 
    old_weight_gradient = old_morpher.calculate_morphing_weight_gradient(theta = [1,1])
    new_weight_gradient = new_morpher.calculate_morphing_weight_gradient(theta = [1,1])

    assert old_weight_gradient.any() == new_weight_gradient .any(), "The weight gradient of old inputs and the new inputs are different."


def evaluate_morphing():
    # Check if the morpher object created with old and new inputs can execute evaluate_morphing() function
    old_morpher = m.PhysicsMorpher(parameter_max_power=[4,4],
                    parameter_range=[(-1.,1.), (-1.,1.)])
    this_components = np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]) 
    this_basis = np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]])

    # Set the powers of components and basis
    old_morpher.set_components(this_components)
    old_morpher.set_basis(basis_numpy= this_basis)

    old_value = old_morpher.evaluate_morphing(basis = this_basis)

    # Morpher object with the new format of inputs
    new_morpher = m.PhysicsMorpher(parameter_max_power=[4,4], parameter_range=[(-1.,1.,), (-1.,1.)])

    gd = None 
    gp = None 
    gs = np.array([[1,1,1,1,1], [-5, -4, -3, -2, -1]])

    _ = new_morpher.find_components(Ns = 2, Nd = 0, Np= 0)
    new_morpher.set_basis(basis_d = gd, basis_p = gp, basis_s=gs)

    new_value = new_morpher.evaluate_morphing(basis = this_basis)

    assert np.abs(new_value - old_value)<10**3, "The output values of old and new evaluate_morphing with same inputs varies more than 10^3"

def test_new_morphing():
    # new input formats, the two should have the same values except for the different input formats
    gd = None 
    gp = None 
    gs = np.array([[1,1,1,1,1], [-5, -4, -3, -2, -1]])

    # old input formats
    parameter_max_power = [2,2]
    this_components = np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]) 
    this_basis = np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]])

    #check if the old and new format output matrices are the same. 
    compare_matrices_weights()

    #check if the new method can work with overdetermined morphing. 
    overdetermined(gs = np.array([[1,1,1,1,1,1,1], [-5, -4, -3, -2, -1, 0, 1]]))

    # Check if the different input format of the same input values will produce the same weight_gradient results.
    weight_gradients(gs = gs, 
                    gp=gp, gd = gd, 
                    parameter_max_power=parameter_max_power, 
                    this_components=this_components, 
                    this_basis=this_basis)

    # Check if the old and new format of inputs can both execute the function
    evaluate_morphing()

    #Check if the find_components method work as expected with several examples
    new_find_components()
    


if __name__=="__main__":
    test_original_mophing()
    test_new_morphing()


