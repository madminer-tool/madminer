# MadMiner

Mining gold from MadGraph to improve limit setting in particle physics. Work in progress by Johann Brehmer, Kyle Cranmer,
and Felix Kling. Note that this is a prototype and all the interfaces are still constantly changing.

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/johannbrehmer/madminer/master)

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

In  [tutorial.ipynb](examples/tutorial/tutorial.ipynb) we provide a detailed tutorial that goes through the main steps
of an analysis.

Note that the suite consists of three packages:
- `madminer` is the core package responsible for the setup of the process, morphing, and the final extraction
  (unweighting) of train and test samples.
- `delphesprocessor` is one example implementation of a detector simulation and observable calculation. This part is
   likely to be swapped out depending on the use case.
- `forge`  contains an implementation of the machine learning part, i.e. trains likelihood ratio estimators on the
  output from the `madminer` package.

## Acknowledgements

The SCANDAL inference method is based on [Masked Autoregressive Flows](https://arxiv.org/abs/1705.07057), and its
implementation is a pyTorch port of the original code by George Papamakarios et al. available at
[https://github.com/gpapamak/maf](https://github.com/gpapamak/maf).

The [setup.py](setup.py) was adapted from
[https://github.com/kennethreitz/setup.py](https://github.com/kennethreitz/setup.py).
