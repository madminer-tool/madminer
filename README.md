# MadMiner: ML based inference for particle physics

**By Johann Brehmer, Felix Kling, Irina Espejo, Sinclert Pérez, and Kyle Cranmer**

[![PyPI version][pypi-version-badge]][pypi-version-link]
[![CI/CD Status][ci-status-badge]][ci-status-link]
[![Docs Status][docs-status-badge]][docs-status-link]
[![Gitter chat][chat-gitter-badge]][chat-gitter-link]
[![Code style][code-style-badge]][code-style-link]
[![MIT license][mit-license-badge]][mit-license-link]
[![DOI reference][ref-zenodo-badge]][ref-zenodo-link]
[![ArXiv reference][ref-arxiv-badge]][ref-arxiv-link]


## Introduction

![Schematics of the simulation and inference workflow][image-rascal-diagram]

Particle physics processes are usually modeled with complex Monte-Carlo simulations of the hard process, parton shower,
and detector interactions. These simulators typically do not admit a tractable likelihood function: given a (potentially
high-dimensional) set of observables, it is usually not possible to calculate the probability of these observables
for some model parameters. Particle physicists usually tackle this problem of "likelihood-free inference" by
hand-picking a few "good" observables or summary statistics and filling histograms of them. But this conventional
approach discards the information in all other observables and often does not scale well to high-dimensional problems.

In the three publications ["Constraining Effective Field Theories with Machine Learning"][ref-arxiv-madminer-1],
["A Guide to Constraining Effective Field Theories with Machine Learning"][ref-arxiv-madminer-2], and
["Mining gold from implicit models to improve likelihood-free inference"][ref-arxiv-madminer-3],
a new approach has been developed. In a nutshell, additional information is extracted from the simulations that is
closely related to the matrix elements that determine the hard process. This "augmented data" can be used to train
neural networks to efficiently approximate arbitrary likelihood ratios. We playfully call this process "mining gold"
from the simulator, since this information may be hard to get, but turns out to be very valuable for inference.

But the gold does not have to be hard to mine: MadMiner automates these modern multivariate inference strategies. It
wraps around the simulators MadGraph and Pythia, with different options for the detector simulation. It streamlines all
steps in the analysis chain from the simulation to the extraction of the augmented data, their processing, the training
and evaluation of the neural networks, and the statistical analysis are implemented.


## Resources

### Paper
Our main publication [MadMiner: Machine-learning-based inference for particle physics][ref-arxiv-link]
provides an overview over this package. We recommend reading it first before jumping into the code.

### Installation instructions
Please have a look at our [installation instructions][docs-installation-guide].

### Tutorials
In the [examples][examples-folder-path] folder in this repository, we provide two tutorials. The first is called
[_Toy simulator_][examples-simulator-path], and it is based on a toy problem rather than a full particle-physics simulation.
It demonstrates inference with MadMiner without spending much time on the more technical steps of running the simulation.
The second, called [_Particle physics_][examples-physics-path], shows all steps of a particle-physics analysis with MadMiner.

These examples are the basis of [the online tutorial][jupyter-tutorial-link] built on Jupyter Books. It also walks
through how to run MadMiner using Docker so that you do not have to install Fortran, MadGraph, Pythia, Delphes, etc.
You can even run it with no install using Binder.

### Documentation
The madminer API is documented on [Read the Docs][docs-index].

### Support
If you have any questions, please chat to us in our [Gitter community][chat-gitter-link].


## Citations

If you use MadMiner, please cite our main publication,
```
@article{Brehmer:2019xox,
      author         = "Brehmer, Johann and Kling, Felix and Espejo, Irina and Cranmer, Kyle",
      title          = "{MadMiner: Machine learning-based inference for particle physics}",
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
      author         = "Brehmer, Johann and Kling, Felix and Espejo, Irina and Perez, Sinclert and Cranmer, Kyle",
      title          = "{MadMiner}",
      doi            = "10.5281/zenodo.1489147",
      url            = {https://github.com/madminer-tool/madminer}
}
```

The main references for the implemented inference techniques are the following:

- CARL: [1506.02169][ref-arxiv-carl].
- MAF: [1705.07057][ref-arxiv-maf].
- CASCAL, RASCAL, ROLR, SALLY, SALLINO, SCANDAL:
  - [1805.00013][ref-arxiv-madminer-1].
  - [1805.00020][ref-arxiv-madminer-2].
  - [1805.12244][ref-arxiv-madminer-3].
- ALICE, ALICES: [1808.00973][ref-arxiv-alice].


## Acknowledgements

We are immensely grateful to all [contributors][repo-madminer-contrib] and bug reporters! In particular, we would like
to thank Zubair Bhatti, Philipp Englert, Lukas Heinrich, Alexander Held, Samuel Homiller and Duccio Pappadopulo.

The SCANDAL inference method is based on [Masked Autoregressive Flows][ref-arxiv-scandal], where our implementation is
a PyTorch port of the original code by George Papamakarios, available at [this repository][repo-maf-main-page].

![IRIS-HEP logo][image-iris-logo]

We are grateful for the support of [IRIS-HEP][web-iris-hep] and [DIANA-HEP][web-diana-hep].


[chat-gitter-badge]: https://badges.gitter.im/madminer/community.svg
[chat-gitter-link]: https://gitter.im/madminer/community
[ci-status-badge]: https://github.com/madminer-tool/madminer/actions/workflows/ci.yml/badge.svg?branch=master
[ci-status-link]: https://github.com/madminer-tool/madminer/actions/workflows/ci.yml?query=branch%3Amaster
[code-style-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[code-style-link]: https://github.com/psf/black
[docs-status-badge]: https://readthedocs.org/projects/madminer/badge/?version=latest
[docs-status-link]: https://madminer.readthedocs.io/en/latest/?badge=latest
[mit-license-badge]: https://img.shields.io/badge/License-MIT-blue.svg
[mit-license-link]: https://github.com/madminer-tool/madminer/blob/master/LICENSE.md
[pypi-version-badge]: https://badge.fury.io/py/madminer.svg
[pypi-version-link]: https://badge.fury.io/py/madminer
[ref-arxiv-badge]: http://img.shields.io/badge/arXiv-1907.10621-B31B1B.svg
[ref-arxiv-link]: https://arxiv.org/abs/1907.10621
[ref-zenodo-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.1489147.svg
[ref-zenodo-link]: https://doi.org/10.5281/zenodo.1489147

[docs-index]: https://madminer.readthedocs.io/en/latest/
[docs-installation-guide ]: https://madminer.readthedocs.io/en/latest/installation.html
[examples-folder-path]: https://github.com/madminer-tool/madminer/tree/master/examples
[examples-physics-path]: https://github.com/madminer-tool/madminer/tree/master/examples/tutorial_particle_physics
[examples-simulator-path]: https://github.com/madminer-tool/madminer/tree/master/examples/tutorial_toy_simulator
[image-iris-logo]: https://iris-hep.org/assets/logos/Iris-hep-4-no-long-name.png
[image-rascal-diagram]: https://raw.githubusercontent.com/madminer-tool/madminer/master/docs/img/rascal-explainer.png
[jupyter-tutorial-link]: https://madminer-tool.github.io/madminer-tutorial
[ref-arxiv-alice]: https://arxiv.org/abs/1808.00973
[ref-arxiv-carl]: https://arxiv.org/abs/1506.02169
[ref-arxiv-maf]: https://arxiv.org/abs/1705.07057
[ref-arxiv-madminer-1]: https://arxiv.org/abs/1805.00013
[ref-arxiv-madminer-2]: https://arxiv.org/abs/1805.00020
[ref-arxiv-madminer-3]: https://arxiv.org/abs/1805.12244
[ref-arxiv-scandal]: https://arxiv.org/abs/1705.07057
[repo-madminer-contrib]: https://github.com/madminer-tool/madminer/graphs/contributors
[repo-maf-main-page]: https://github.com/gpapamak/maf
[web-diana-hep]: https://diana-hep.org
[web-iris-hep]: https://iris-hep.org
