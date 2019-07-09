# Getting started

## Simulator dependencies

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
  
Finally, Delphes can be replaced with another detector simulation, for instance a full detector simulation based
with Geant4. In this case, the user has to implement code that runs the detector simulation, calculates the observables,
and stores the observables and weights in the HDF5 file. The `DelphesProcessor` and `LHEProcessor` classes might provide
some guidance for this.

We're currently working on a [reference Docker image](https://hub.docker.com/r/irinahub/docker-madminer-madgraph) that
has all these dependencies and the needed patches installed.

## Install MadMiner

To install the MadMiner package with all its Python dependencies, run `pip install madminer`.

To get the latest development version as well as the tutorials, clone the
[GitHub repository](https://github.com/johannbrehmer/madminer) and run `pip install -e .` from the repository main
folder.

### Docker image

At [https://hub.docker.com/u/madminertool/](https://hub.docker.com/u/madminertool/) we provide Docker images for
the latest version of MadMiner and the physics simulator. Please email [iem244@nyu.edu](iem244@nyu.edu) for any
questions about the Docker images.
