# MadMiner

Mining gold from MadGraph to improve limit setting in particle physics. Work in progress by Johann Brehmer, Kyle Cranmer,
and Felix Kling. Note that this is a prototype and all the interfaces are still constantly changing.

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/johannbrehmer/madminer/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

## Introduction

For an introduction to the implemented inference methods, see
["Constraining Effective Field Theories With Machine Learning"](https://arxiv.org/abs/1805.00013) and
["A Guide to Constraining Effective Field Theories With Machine Learning"](https://arxiv.org/abs/1805.00020),
both by Johann Brehmer, Gilles Louppe, Juan Pavez, and Kyle Cranmer.

## Getting started

### Core dependencies

Make sure the following dependencies are installed and running:
- MadGraph (we've tested our setup with MG5_aMC v2.6.2 and have received reports about issues with newer versions)
- Pythia8 and the MG-Pythia interface installed from the MadGraph interface. The MadGraph-Pythia interface has issues
with the treatment of multiple weights. Until this is fixed in the official release, the user has to install a patch
manually. These files are available upon request.
- Python packages as given in [environment.yml](environment.yml). You can create a conda environment from this file, see
(https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

### Detector simulation and observable calculation

First, we provide an option to calculate truth-level observables which do not require any additional packages.

We have also implemented a fast detector simulation and observable calculation in our workflow. This adds another
requirement:
- Delphes, for instance installed from the MadGraph interface. Delphes has issues with the treatment of multiple
weights. Until this is fixed in the official releases, the user has to install a patch manually. These files are also
available upon request.

Finally, Delphes can be replaced with another detector simulation, for instance a full detector simulation based
with Geant4. In this case, the user has to implement code that runs the detector simulation, calculates the observables,
and stores the observables and weights in the HDF5 file. The `DelphesProcessor` and `LHEProcessor` classes might provide
some guidance for this.

### Installation

Clone the repository and run `pip install -e <path to repository>`.

## Using MadMiner

### Tutorials

In [tutorial_1.ipynb](examples/tutorial/tutorial_1.ipynb) we provide a detailed tutorial that goes through the main
steps of a detector-level analysis.

After that, we recommend going through [tutorial_2.ipynb](examples/tutorial/tutorial_1.ipynb), which explains local
score methods, how to estimate the Fisher information, and introduces some convenient ensemble methods.

Finally, [tutorial_parton.ipynb](examples/tutorial/tutorial.ipynb) explains how to perform a parton-level Fisher
information analysis.

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

In [docs](docs/) we provide a HTML and PDF documentation of the different modules and classes.

## Acknowledgements

The SCANDAL inference method is based on [Masked Autoregressive Flows](https://arxiv.org/abs/1705.07057), and its
implementation is a pyTorch port of the original code by George Papamakarios et al. available at
[https://github.com/gpapamak/maf](https://github.com/gpapamak/maf).

The [setup.py](setup.py) was adapted from
[https://github.com/kennethreitz/setup.py](https://github.com/kennethreitz/setup.py).

## References

General method papers:
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

Physics publications:
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

Individual inference methods are introduced in the following papers:
- CARL:
```
@article{Cranmer:2015bka,
      author         = "Cranmer, Kyle and Pavez, Juan and Louppe, Gilles",
      title          = "{Approximating Likelihood Ratios with Calibrated
                        Discriminative  Classifiers}",
      year           = "2015",
      eprint         = "1506.02169",
      archivePrefix  = "arXiv",
      primaryClass   = "stat.AP",
}
```
- Masked Autoregressive Flows:
```
@incollection{2017arXiv170507057P,
      title = {Masked Autoregressive Flow for Density Estimation},
      author = {Papamakarios, George and Murray, Iain and Pavlakou, Theo},
      booktitle = {Advances in Neural Information Processing Systems 30},
      editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
      pages = {2338--2347},
      year = {2017},
}
```
- ALICE: 
```
@article{Stoye:2018ovl,
      author         = "Stoye, Markus and Brehmer, Johann and Louppe, Gilles and
                        Pavez, Juan and Cranmer, Kyle",
      title          = "{Likelihood-free inference with an improved cross-entropy
                        estimator}",
      year           = "2018",
      eprint         = "1808.00973",
      archivePrefix  = "arXiv",
      primaryClass   = "stat.ML",
      SLACcitation   = "%%CITATION = ARXIV:1808.00973;%%"
}
```