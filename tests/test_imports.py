from madminer.core import MadMiner
from madminer.delphes import DelphesProcessor
from madminer.lhe import LHEProcessor
from madminer.ml import EnsembleForge, MLForge
from madminer.morphing import Morpher, NuisanceMorpher
from madminer.plotting import plot_2d_morphing_basis, plot_distribution_of_information, plot_distributions
from madminer.plotting import plot_fisher_information_contours_2d, plot_fisherinfo_barplot
from madminer.plotting import plot_nd_morphing_basis_scatter, plot_2d_morphing_basis
from madminer.sampling import SampleAugmenter


def test_imports():
    assert True
