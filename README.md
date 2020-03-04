# MadMiner: Machine learning–based inference for particle physics

**By Johann Brehmer, Felix Kling, Irina Espejo, and Kyle Cranmer**

[![PyPI version](https://badge.fury.io/py/madminer.svg)](https://badge.fury.io/py/madminer)
[![Build Status](https://travis-ci.com/diana-hep/madminer.svg?branch=master)](https://travis-ci.com/diana-hep/madminer)
[![Documentation Status](https://readthedocs.org/projects/madminer/badge/?version=latest)](https://madminer.readthedocs.io/en/latest/?badge=latest)
[![Gitter](https://badges.gitter.im/madminer/community.svg)](https://gitter.im/madminer/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1489147.svg)](https://doi.org/10.5281/zenodo.1489147)
[![arXiv](http://img.shields.io/badge/arXiv-1907.10621-B31B1B.svg)](https://arxiv.org/abs/1907.10621)


## Introduction

Particle physics processes are usually modeled with complex Monte-Carlo simulations of the hard process, parton shower,
and detector interactions. These simulators typically do not admit a tractable likelihood function: given a (potentially
high-dimensional) set of observables, it is usually not possible to calculate the probability of these observables
for some model parameters. Particle physicisists usually tackle this problem of "likelihood-free inference" by
hand-picking a few "good" observables or summary statistics and filling histograms of them. But this conventional
approach discards the information in all other observables and often does not scale well to high-dimensional problems.

In the three publications
["Constraining Effective Field Theories With Machine Learning"](https://arxiv.org/abs/1805.00013),
["A Guide to Constraining Effective Field Theories With Machine Learning"](https://arxiv.org/abs/1805.00020), and
["Mining gold from implicit models to improve likelihood-free inference"](https://arxiv.org/abs/1805.12244),
a new approach has been developed. In a nutshell, additional information is extracted from the simulations that is
closely related to the matrix elements that determine the hard process. This
"augmented data" can be used to train neural networks to efficiently approximate arbitrary likelihood ratios. We
playfully call this process "mining gold" from the simulator, since this information may be hard to get, but turns out
to be very valuable for inference.

But the gold does not have to be hard to mine: MadMiner automates these modern multivariate inference strategies. It
wraps around the simulators MadGraph and Pythia, with different options for the detector simulation. It streamlines all
steps in the analysis chain from the simulation to the extraction of the augmented data, their processing, the training
and evaluation of the neural networks, and the statistical analysis are implemented.


## Resources


### Paper

Our main publication [MadMiner: Machine-learning-based inference for particle physics](https://arxiv.org/abs/1907.10621)
provides an overview over this package. We recommend reading it first before jumping into the code.


### Installation instructions

Please have a look at our [installation instructions](https://madminer.readthedocs.io/en/latest/installation.html).


### Tutorials

In the [examples](examples/) folder in this repository, we provide two tutorials. The first at
[examples/tutorial_toy_simulator/tutorial_toy_simulator.ipynb](examples/tutorial_toy_simulator/tutorial_toy_simulator.ipynb)
is based on a toy problem rather than a full particle-physics simulation. It demonstrates
inference with MadMiner without spending much time on the more technical steps of running the simulation. The second,
at [examples/tutorial_particle_physics](examples/tutorial_particle_physics), shows all steps of a particle-physics
analysis with MadMiner.

These examples are the basis of [an online tutorial](https://cranmer.github.io/madminer-tutorial/intro) built with on Jupyter Books. It also walks through how to run MadMiner using docker so that you don't have to install Fortran, MadGraph, Pythia, Delphes, etc. You can even run it with no install using binder. 

### Documentation

The madminer API is documented on [readthedocs](https://madminer.readthedocs.io/en/latest/?badge=latest).


### Support

If you have any questions, please
chat to us [in our Gitter community](https://gitter.im/madminer/community) or write us at 
[johann.brehmer@nyu.edu](johann.brehmer@nyu.edu).


## Citations

If you use MadMiner, please cite our main publication,
```
@article{Brehmer:2019xox,
      author         = "Brehmer, Johann and Kling, Felix and Espejo, Irina and
                        Cranmer, Kyle",
      title          = "{MadMiner: Machine learning-based inference for particle
                        physics}",
      journal        = "Comput. Softw. Big Sci.",
      volume         = "4",
      year           = "2020",
      number         = "1",
      pages          = "3",
      doi            = "10.1007/s41781-020-0035-2",
      eprint         = "1907.10621",
      archivePrefix  = "arXiv",
      primaryClass   = "hep-ph",
      SLACcitation   = "%%CITATION = ARXIV:1907.10621;%%"
}
```

The code itself can be cited as
```
@misc{MadMiner_code,
      author         = "Brehmer, Johann and Kling, Felix and Espejo, Irina and Cranmer, Kyle",
      title          = "{MadMiner}",
      doi            = "10.5281/zenodo.1489147",
      url            = {https://github.com/diana-hep/madminer}
}
```

The main references for the implemented inference techniques are the following:

- CARL: [1506.02169](https://arxiv.org/abs/1506.02169)
- MAF: [1705.07057](https://arxiv.org/abs/1705.07057)
- CASCAL, RASCAL, ROLR, SALLY, SALLINO, SCANDAL: [1805.00013](https://arxiv.org/abs/1805.00013), [1805.00020](https://arxiv.org/abs/1805.00020), [1805.12244](https://arxiv.org/abs/1805.12244)
- ALICE, ALICES: [1808.00973](https://arxiv.org/abs/1808.00973)


## Acknowledgements

We are immensely grateful to all contributors and bug reporters! In particular, we would like to thank Zubair Bhatti,
Philipp Englert, Lukas Heinrich, Alexander Held, Samuel Homiller, and Duccio Pappadopulo.

The SCANDAL inference method is based on [Masked Autoregressive Flows](https://arxiv.org/abs/1705.07057), and our
implementation is a pyTorch port of the original code by George Papamakarios et al., which is available at
[https://github.com/gpapamak/maf](https://github.com/gpapamak/maf).

The [setup.py](setup.py) was adapted from
[https://github.com/kennethreitz/setup.py](https://github.com/kennethreitz/setup.py).
