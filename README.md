# MadMiner: An inference toolkit for particle physics

**Johann Brehmer, Felix Kling, Irina Espejo, and Kyle Cranmer**

[![PyPI version](https://badge.fury.io/py/madminer.svg)](https://badge.fury.io/py/madminer)
[![Documentation Status](https://readthedocs.org/projects/madminer/badge/?version=latest)](https://madminer.readthedocs.io/en/latest/?badge=latest)
[![Gitter](https://badges.gitter.im/madminer/community.svg)](https://gitter.im/madminer/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Build Status](https://travis-ci.com/diana-hep/madminer.svg?branch=master)](https://travis-ci.com/diana-hep/madminer)
[![Docker pulls](https://img.shields.io/docker/pulls/madminertool/docker-madminer.svg)](https://hub.docker.com/r/madminertool/docker-madminer)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/johannbrehmer/madminer/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1489147.svg)](https://doi.org/10.5281/zenodo.1489147)

*Note that this is a development version. Do not expect anything to be stable. If you have any questions, please
chat to us [in our Gitter community](https://gitter.im/madminer/community) or write us at 
[johann.brehmer@nyu.edu](johann.brehmer@nyu.edu).*

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
- MadGraph (we've tested our setup with MG5_aMC v2.6.2 and v2.6.5). See
  [https://launchpad.net/mg5amcnlo](https://launchpad.net/mg5amcnlo) for installation instructions. Note that MadGraph
  requires a Fortran compiler as well as Python 2.6 or 2.7. (Note that you can still run most MadMiner analysis steps
  with Python 3.)
- For the analysis of systematic uncertainties, LHAPDF6 has to be installed with Python support (see also
  [the documentation of MadGraph's systematics tool](https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/Systematics)).

For the detector simulation part, there are different options. For simple parton-level analyses, we provide a bare-bones
option to calculate truth-level observables which do not require any additional packages.

We have also implemented a fast detector simulation based on Delphes with a flexible framework to calculate observables.
Using this adds additional requirements:
- Pythia8 and the MG-Pythia interface, installed from within the MadGraph command line interface: execute
 `<MadGraph5_directory>/bin/mg5_aMC`, and then inside the MadGraph interface, run `install pythia8` and
 `install mg5amc_py8_interface`.
- Delphes. Again, you can (but this time you don't have to) install it from the MadGraph command line interface with
  `install Delphes`.
  
*(These tools currently have a bug: the MG-Pythia interface and Delphes currently do not keep track of additional weights
that are in the LHE file. This is not a big deal, MadMiner now offers an option to extract these weights from the
LHE file. Alternatively, there is a unofficial patch for these tools that solves these issues. It is available upon
request.)*

Finally, Delphes can be replaced with another detector simulation, for instance a full detector simulation based
with Geant4. In this case, the user has to implement code that runs the detector simulation, calculates the observables,
and stores the observables and weights in the HDF5 file. The `DelphesProcessor` and `LHEProcessor` classes might provide
some guidance for this.

### Install MadMiner

To install the MadMiner package with all its Python dependencies, run `pip install madminer`.

To get the [examples](examples/), including the tutorials, clone this repository.

### Docker image

At [https://hub.docker.com/u/madminertool/](https://hub.docker.com/u/madminertool/) we provide Docker images for
the latest version of MadMiner and the physics simulator. Please email [iem244@nyu.edu](iem244@nyu.edu) for any
questions about the Docker images.

## Using MadMiner

### Tutorials

As a starting point, we recommend to look at a 
[tutorial based on a toy example](examples/tutorial_toy_simulator/tutorial_toy_simulator.ipynb). It demonstrates
inference with MadMinier without spending much time on the more technical steps of running the simulation.

We then provide two sets of tutorials for the same real-world particle physics process. The difference between them is
that the [parton-level tutorial](examples/tutorial_parton_level/) only requires running MadGraph. Instead of a proper
shower and detector simulation, we describe detector effects through simple smearing functions. This reduces the runtime
of the scripts quite a bit. In the [Delphes tutorial](examples/tutorial_delphes), we finally switch to Pythia and
Delphes; this tutorial is probably best suited as a starting point for phenomenological research projects. In most
other aspects, the two tutorials are identical.

[Other provided examples](examples/) show MadMiner in action in different processes.

### Documentation

The madminer API is documented on [readthedocs](https://madminer.readthedocs.io/en/latest/?badge=latest), including
a walk-through through the code.


## Acknowledgements

We are immensely grateful to all contributors and bug reporters! In particular, we would like to thank Zubair Bhatti,
Alexander Held, and Duccio Pappadopulo. A big thanks to Lukas Heinrich for his help with workflows and Docker
containers.

The SCANDAL inference method is based on [Masked Autoregressive Flows](https://arxiv.org/abs/1705.07057), and our
implementation is a pyTorch port of the original code by George Papamakarios et al., which is available at
[https://github.com/gpapamak/maf](https://github.com/gpapamak/maf).

The [setup.py](setup.py) was adapted from
[https://github.com/kennethreitz/setup.py](https://github.com/kennethreitz/setup.py).

## References

If you use MadMiner, please cite this code as
```
@misc{MadMiner,
      author         = "Brehmer, Johann and Kling, Felix and Espejo, Irina and Cranmer, Kyle",
      title          = "{MadMiner}",
      doi            = "10.5281/zenodo.1489147",
      url            = {https://github.com/johannbrehmer/madminer}
}
```

For the inference methods, there are three main references. Two introduce most of the methods in a particle physics
setting:
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

In addition, the inference techniques are discussed in a more general setting, and the SCANDAL family of methods is
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

Some inference methods are introduced in other papers, including [CARL](https://arxiv.org/abs/1506.02169),
[Masked Autoregressive Flows](https://arxiv.org/abs/1705.07057), and [ALICE(S)](https://arxiv.org/abs/1808.00973).
