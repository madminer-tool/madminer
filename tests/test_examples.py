import papermill
import pytest
import tempfile

from pathlib import Path


MORPHING_EXAMPLES_PATH = Path("examples/morphing_basis")
TUTORIAL_EXAMPLES_PATH = Path("examples/tutorial_particle_physics")
SIMULATOR_EXAMPLES_PATH = Path("examples/tutorial_toy_simulator")


@pytest.fixture()
def common_kwargs():
    temp_dir = Path(tempfile.gettempdir())
    return {
        "output_path": temp_dir.joinpath("output.ipynb"),
        "kernel_name": "Python3",
    }


@pytest.mark.skip
@pytest.mark.notebook
def test_morphing_animation(common_kwargs):
    papermill.execute_notebook(
        input_path=MORPHING_EXAMPLES_PATH.joinpath("animate.ipynb"),
        cwd=MORPHING_EXAMPLES_PATH,
        **common_kwargs,
    )


@pytest.mark.skip
@pytest.mark.notebook
def test_morphing_basis(common_kwargs):
    papermill.execute_notebook(
        input_path=MORPHING_EXAMPLES_PATH.joinpath("interactive_basis_chooser.ipynb"),
        cwd=MORPHING_EXAMPLES_PATH,
        **common_kwargs,
    )


@pytest.mark.notebook
def test_physics_tutorial(common_kwargs):
    papermill.execute_notebook(
        input_path=TUTORIAL_EXAMPLES_PATH.joinpath("1_setup.ipynb"),
        cwd=TUTORIAL_EXAMPLES_PATH,
        **common_kwargs,
    )
    papermill.execute_notebook(
        input_path=TUTORIAL_EXAMPLES_PATH.joinpath("2a_parton_level_analysis.ipynb"),
        cwd=TUTORIAL_EXAMPLES_PATH,
        **common_kwargs,
    )
    papermill.execute_notebook(
        input_path=TUTORIAL_EXAMPLES_PATH.joinpath("2b_delphes_level_analysis.ipynb"),
        cwd=TUTORIAL_EXAMPLES_PATH,
        **common_kwargs,
    )
    papermill.execute_notebook(
        input_path=TUTORIAL_EXAMPLES_PATH.joinpath("3a_likelihood_ratio.ipynb"),
        cwd=TUTORIAL_EXAMPLES_PATH,
        **common_kwargs,
    )
    papermill.execute_notebook(
        input_path=TUTORIAL_EXAMPLES_PATH.joinpath("3b_score.ipynb"),
        cwd=TUTORIAL_EXAMPLES_PATH,
        **common_kwargs,
    )
    papermill.execute_notebook(
        input_path=TUTORIAL_EXAMPLES_PATH.joinpath("3c_likelihood.ipynb"),
        cwd=TUTORIAL_EXAMPLES_PATH,
        **common_kwargs,
    )
    papermill.execute_notebook(
        input_path=TUTORIAL_EXAMPLES_PATH.joinpath("4a_limits.ipynb"),
        cwd=TUTORIAL_EXAMPLES_PATH,
        **common_kwargs,
    )
    papermill.execute_notebook(
        input_path=TUTORIAL_EXAMPLES_PATH.joinpath("4b_fisher_information.ipynb"),
        cwd=TUTORIAL_EXAMPLES_PATH,
        **common_kwargs,
    )
    papermill.execute_notebook(
        input_path=TUTORIAL_EXAMPLES_PATH.joinpath("4c_information_geometry.ipynb"),
        cwd=TUTORIAL_EXAMPLES_PATH,
        **common_kwargs,
    )


@pytest.mark.skip
@pytest.mark.notebook
def test_physics_appendix_1(common_kwargs):
    papermill.execute_notebook(
        input_path=TUTORIAL_EXAMPLES_PATH.joinpath("A1_systematic_uncertainties.ipynb"),
        cwd=TUTORIAL_EXAMPLES_PATH,
        **common_kwargs,
    )


@pytest.mark.skip
@pytest.mark.notebook
def test_physics_appendix_2(common_kwargs):
    papermill.execute_notebook(
        input_path=TUTORIAL_EXAMPLES_PATH.joinpath("A2_ensemble_methods.ipynb"),
        cwd=TUTORIAL_EXAMPLES_PATH,
        **common_kwargs,
    )


@pytest.mark.skip
@pytest.mark.notebook
def test_physics_appendix_3(common_kwargs):
    papermill.execute_notebook(
        input_path=TUTORIAL_EXAMPLES_PATH.joinpath("A3_reweighting_existing_samples.ipynb"),
        cwd=TUTORIAL_EXAMPLES_PATH,
        **common_kwargs,
    )


@pytest.mark.skip
@pytest.mark.notebook
def test_physics_appendix_4(common_kwargs):
    papermill.execute_notebook(
        input_path=TUTORIAL_EXAMPLES_PATH.joinpath("A4_lh_nosyst.ipynb"),
        cwd=TUTORIAL_EXAMPLES_PATH,
        **common_kwargs,
    )


@pytest.mark.skip
@pytest.mark.notebook
def test_physics_appendix_5(common_kwargs):
    papermill.execute_notebook(
        input_path=TUTORIAL_EXAMPLES_PATH.joinpath("A5_test_new_likelihood_module.ipynb"),
        cwd=TUTORIAL_EXAMPLES_PATH,
        **common_kwargs,
    )


@pytest.mark.skip
@pytest.mark.notebook
def test_physics_appendix_6(common_kwargs):
    papermill.execute_notebook(
        input_path=TUTORIAL_EXAMPLES_PATH.joinpath("A6_finite_differences.ipynb"),
        cwd=TUTORIAL_EXAMPLES_PATH,
        **common_kwargs,
    )


@pytest.mark.skip
@pytest.mark.notebook
def test_toy_simulator(common_kwargs):
    papermill.execute_notebook(
        input_path=SIMULATOR_EXAMPLES_PATH.joinpath("tutorial_toy_simulator.ipynb"),
        cwd=SIMULATOR_EXAMPLES_PATH,
        **common_kwargs,
    )
