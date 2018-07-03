# MadMiner

Mining gold from MadGraph to improve limit setting in particle physics. Work in progress by Johann Brehmer, Kyle Cranmer,
and Felix Kling. Note that this is a prototype and all the interfaces are still constantly changing.

## Introduction

See ["Mining gold from implicit models to improve likelihood-free inference"](https://arxiv.org/abs/1805.12244) by
Johann Brehmer, Gilles Louppe, Juan Pavez, and Kyle Cranmer.

## Preparation

### Prerequisites

Core dependencies:
- MadGraph interfaced to Pythia 8
- standard packages as given in [environment.yml](environment.yml)

For a simple, automatized detector simulation and observable calculation:
- Delphes
- [DelphesMiner](https://github.com/johannbrehmer/delphesminer)

If Delphes and Delphes miner are not used, the user has to take care of the detector simulation and extraction of observables themselves.

The MadGraph-Pythia interface and Delphes have issues with the treatment of multiple weights. Until this is fixed
in the official releases, the user has to install patches manually.  These patches are available upon request.

### Installation

Clone repository and make sure it is in the PYTHONPATH.

## Usage

See [example.ipynb](examples/usage/example.ipynb).
