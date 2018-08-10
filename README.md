# MadMiner

Mining gold from MadGraph to improve limit setting in particle physics. Work in progress by Johann Brehmer, Kyle Cranmer,
and Felix Kling. Note that this is a prototype and all the interfaces are still constantly changing.

## Introduction

See ["Constraining Effective Field Theories With Machine Learning"](https://arxiv.org/abs/1805.00013) and
["A Guide to Constraining Effective Field Theories With Machine Learning"](https://arxiv.org/abs/1805.00020), both by
Johann Brehmer, Gilles Louppe, Juan Pavez, and Kyle Cranmer.

## Preparation

### Core dependencies

Core dependencies:
- MadGraph interfaced to Pythia 8
- Python packages as given in [environment.yml](environment.yml)

The MadGraph-Pythia interface has issues with the treatment of multiple weights. Until this is fixed
in the official releases, the user has to install patches manually.  These patches are available upon request.

### Detector simulation

For an automized fast detector simulation and observable calculation:
- Delphes
- Python packages as given in [environment.yml](environment.yml)

Delphes has issues with the treatment of multiple weights. Until this is fixed in the official releases, the user has to install a patch
manually. This patch is available upon request.

If Delphes and Delphes miner are not used, the user has to take care of the detector simulation and extraction of observables themselves.

Delphes also has issues with the treatment of multiple weights. Until this is fixed in the official releases, the user
has to install patches (available upon request) manually.

### Installation

Clone repository and make sure it is in the PYTHONPATH.

## Tutorial

See [tutorial.ipynb](examples/tutorial/tutorial.ipynb).

Note that the suite consists of three packages:
- `madminer` is the core package responsible for the setup of the process, morphing, and the final extraction
  (unweighting) of train and test samples.
- `delphesprocessor` is one example implementation of a detector simulation and observable calculation. This part is
   likely to swapped out depending on the use case.
- `forge`  contains an implementation of the machine learning part, i.e. trains likelihood ratio estimators on the
  output from the `madminer` package.
