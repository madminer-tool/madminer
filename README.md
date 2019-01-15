# MadMiner

*Johann Brehmer, Felix Kling, and Kyle Cranmer*

Mining gold from MadGraph to improve limit setting in particle physics.

Note that this is an early development version. Do not expect anything to be stable. If you have any questions, please
contact us at [johann.brehmer@nyu.edu](johann.brehmer@nyu.edu).

[![PyPI version](https://badge.fury.io/py/madminer.svg)](https://badge.fury.io/py/madminer)
[![Documentation Status](https://readthedocs.org/projects/madminer/badge/?version=latest)](https://madminer.readthedocs.io/en/latest/?badge=latest)
[![Docker pulls](https://img.shields.io/docker/pulls/irinahub/docker-madminer-madgraph.svg)](https://hub.docker.com/r/irinahub/docker-madminer-madgraph)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/johannbrehmer/madminer/master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1489147.svg)](https://doi.org/10.5281/zenodo.1489147)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


## Introduction

Particle physics processes are usually modelled with complex Monte-Carlo simulations of the hard process, parton shower,
and detector interactions. These simulators typically do not admit a tractable likelihood function: given a (potentially
high-dimensional) set of observables, it is usually not possible to calculate the probability of these observables
for some model parameters. Particle physicisists usually tackle this problem of "likelihood-free inference" by
hand-picking a few "good" observables or summary statistics and filling histograms of them. But this conventional
approach discards the information in all other observables and often does not scale well to high-dimensional problems.

In the three publications
["Constraining Effective Field Theories With Machine Learning"](https://arxiv.org/abs/1805.00013),
["A Guide to Constraining Effective Field Theories With Machine Learning"](https://arxiv.org/abs/1805.00020), and
["Mining gold from implicit models to improve likelihood-free inference"](https://arxiv.org/abs/1805.12244),
a new approach has been developed. In a nut shell, additional information is extracted from the simulators. This
"augmented data" can be used to train neural networks to efficiently approximate arbitrary likelihood ratios. We
playfully call this process "mining gold" from the simulator, since this information may be hard to get, but turns out
to be very valuable for inference.

But the gold does not have to be hard to mine. This package automates these inference strategies. It wraps around the
simulators MadGraph and Pythia, with different options for the detector simulation. All steps in the analysis chain from
the simulation to the extraction of the augmented data, their processing, and the training and evaluation of the neural
estimators are implemented.

## Getting started

### Simulator dependencies

Make sure the following tools are installed and running:
- MadGraph (we've tested our setup with MG5_aMC v2.6.2 and have received reports about issues with newer versions).
- Pythia8 and the MG-Pythia interface installed from the MadGraph interface.
- For the analysis of systematic uncertainties, LHAPDF6 has to be installed with Python support (see also
[the documentation of MadGraph's systematics tool](https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/Systematics)).

For the detector simulation part, there are different options. For simple parton-level analyses, we provide a bare-bones
option to calculate truth-level observables which do not require any additional packages.

We have also implemented a fast detector simulation based on Delphes with a flexible framework to calculate observables.
Using this adds another requirement:
- Delphes, for instance installed from the MadGraph interface.

Finally, Delphes can be replaced with another detector simulation, for instance a full detector simulation based
with Geant4. In this case, the user has to implement code that runs the detector simulation, calculates the observables,
and stores the observables and weights in the HDF5 file. The `DelphesProcessor` and `LHEProcessor` classes might provide
some guidance for this.

We're currently working on a [reference Docker image](https://hub.docker.com/r/irinahub/docker-madminer-madgraph) that
has all these dependencies and the needed patches installed.

### Install MadMiner

To install the MadMiner package with all its Python dependencies, run `pip install madminer`.

To get the [examples](examples/), including the tutorials, clone this repository.

## Using MadMiner

### Tutorials

As a starting point, we recommend the [toy example](examples/toy_example/toy_example.ipynb), which demonstrates
inference with MadMinier without spending much time on the more technical steps of running the simulation.

In [tutorial_1.ipynb](examples/tutorial/tutorial_1.ipynb) we provide a detailed tutorial that goes through the main
steps of a detector-level analysis.

After that, we recommend going through [tutorial_2.ipynb](examples/tutorial/tutorial_1.ipynb), which explains local
score methods, how to estimate the Fisher information, and introduces some convenient ensemble methods.

[Other provided examples](examples/) show MadMiner in action in different processes.

### Package structure

- `madminer.core` contains the functions to set up the process, parameter space, morphing, and to steer MadGraph and
   Pythia.
- `madminer.delphes` and `madminer.lhe` contain two example implementations of a detector simulation and observable
   calculation. This part can easily be swapped out depending on the use case.
- In `madminer.sampling`, train and test samples for the machine learning part are generated and augmented with the
  joint score and joint ratio.
- `madminer.ml`  contains an implementation of the machine learning part. The user can train and evaluate estimators
  for the likelihood ratio or score.
- Finally,  `madminer.fisherinformation` contains functions to calculate the Fisher information, both on parton level
  or detector level, in the full process, individual observables, or the total cross section.

### Documentation

The madminer API is documented on [readthedocs](https://madminer.readthedocs.io/en/latest/?badge=latest).

## Acknowledgements

The SCANDAL inference method is based on [Masked Autoregressive Flows](https://arxiv.org/abs/1705.07057), and its
implementation is a pyTorch port of the original code by George Papamakarios et al. available at
[https://github.com/gpapamak/maf](https://github.com/gpapamak/maf).

The [setup.py](setup.py) was adapted from
[https://github.com/kennethreitz/setup.py](https://github.com/kennethreitz/setup.py).

## References

There are two main references for these inference methods applied to problems in particle physics:
```
@article{Brehmer:2018kdj,
      author         = "Brehmer, Johann and Cranmer, Kyle and Louppe, Gilles and
                        Pavez, Juan",
      title          = "{Constraining Effective Field Theories with Machine
                        Learning}",
      journal        = "Phys. Rev. Lett.",
      volume         = "121",
      year           = "2018",
      number         = "11",
      pages          = "111801",
      doi            = "10.1103/PhysRevLett.121.111801",
      eprint         = "1805.00013",
      archivePrefix  = "arXiv",
      primaryClass   = "hep-ph",
}

@article{Brehmer:2018eca,
      author         = "Brehmer, Johann and Cranmer, Kyle and Louppe, Gilles and
                        Pavez, Juan",
      title          = "{A Guide to Constraining Effective Field Theories with
                        Machine Learning}",
      journal        = "Phys. Rev.",
      volume         = "D98",
      year           = "2018",
      number         = "5",
      pages          = "052004",
      doi            = "10.1103/PhysRevD.98.052004",
      eprint         = "1805.00020",
      archivePrefix  = "arXiv",
      primaryClass   = "hep-ph",
}
```

In addition, the inference techniques are discussed in a more abstract setting, and the SCANDAL family of methods is
added in:
```
@article{Brehmer:2018hga,
      author         = "Brehmer, Johann and Louppe, Gilles and Pavez, Juan and
                        Cranmer, Kyle",
      title          = "{Mining gold from implicit models to improve
                        likelihood-free inference}",
      year           = "2018",
      eprint         = "1805.12244",
      archivePrefix  = "arXiv",
      primaryClass   = "stat.ML",
      SLACcitation   = "%%CITATION = ARXIV:1805.12244;%%"
}
```

Individual inference methods are introduced in other papers, including [CARL](https://arxiv.org/abs/1506.02169),
[Masked Autoregressive Flows](https://arxiv.org/abs/1705.07057), and [ALICE(S)](https://arxiv.org/abs/1808.00973).
