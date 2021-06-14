# Using MadMiner

We provide different resources that help with the use of MadMiner:


## Paper

Our main publication [MadMiner: Machine-learning-based inference for particle physics](https://arxiv.org/abs/1907.10621)
provides an overview over this package. We recommend reading it first before jumping into the code.


## Tutorials

In the [examples](https://github.com/diana-hep/madminer/tree/master/examples) folder in the MadMiner repository, we
provide two tutorials. The first at
[examples/tutorial_toy_simulator/tutorial_toy_simulator.ipynb](https://github.com/diana-hep/madminer/blob/master/examples/tutorial_toy_simulator/tutorial_toy_simulator.ipynb)
is based on a toy problem rather than a full particle-physics simulation. It demonstrates
inference with MadMiner without spending much time on the more technical steps of running the simulation. The second, at
[examples/tutorial_particle_physics](https://github.com/diana-hep/madminer/tree/master/examples/tutorial_particle_physics),
shows all steps of a particle-physics analysis with MadMiner.


## Typical work flow

Here we illustrate the structure of data analysis with MadMiner:

![MadMiner workflow](img/workflow_combined.jpg)

- `madminer.core` contains the functions to set up the process, parameter space, morphing, and to steer MadGraph and
   Pythia.
- `madminer.lhe` and `madminer.delphes` contain two example implementations of a detector simulation and observable
   calculation. This part can easily be swapped out depending on the use case.
- In `madminer.sampling`, train and test samples for the machine learning part are generated and augmented with the
  joint score and joint ratio.
- `madminer.ml`  contains an implementation of the machine learning part. The user can train and evaluate estimators
  for the likelihood ratio or score.
- Finally,  `madminer.fisherinformation` contains functions to calculate the Fisher information, both on parton level
  or detector level, in the full process, individual observables, or the total cross section.


## Technical documentation

The madminer API is documented on here as well, just look through the pages linked on the left.


## Support

If you have any questions, please
chat to us [in our Gitter community](https://gitter.im/madminer/community) or write us at 
[johann.brehmer@nyu.edu](mailto:johann.brehmer@nyu.edu).
