# Getting started

## Simulator dependencies

Make sure the following tools are installed and running:
- MadGraph (we have tested our setup with version 2.8.0+). See [MadGraph's website][web-madgraph-main-page]
  for installation instructions. Note that MadGraph requires a Fortran compiler as well as Python 3.6+.
- For the analysis of systematic uncertainties, LHAPDF6 has to be installed with Python support
  (see also [the documentation of MadGraph's systematics tool][web-madgraph-systematics]).

For the detector simulation part, there are different options. For simple parton-level analyses, we provide a bare-bones
option to calculate truth-level observables which do not require any additional packages. We have also implemented
a fast detector simulation based on Delphes with a flexible framework to calculate observables. 
Using this adds additional requirements:

```shell
echo "install pythia8" | python3 <MadGraph_dir>/bin/mg5_aMC
echo "install Delphes" | python3 <MadGraph_dir>/bin/mg5_aMC
```

Finally, Delphes can be replaced with another detector simulation, for instance a full detector simulation based
with Geant4. In this case, the user has to implement code that runs the detector simulation, calculates the observables,
and stores the observables and weights in the HDF5 file. The `DelphesProcessor` and `LHEProcessor` classes might provide
some guidance for this.


## Install MadMiner

To install the MadMiner package with all its Python dependencies, run `pip install madminer`.

To get the latest development version as well as the tutorials, clone the [GitHub repository][repo-madminer]
and run `pip install -e .` from the repository main folder.


## Docker image

At the DockerHub [madminertool organization][docker-madminer] we provide Docker images for the latest version of MadMiner.
Please email [Irina Espejo](mailto:iem244@nyu.edu) for any questions about the Docker images.


[docker-madminer]: https://hub.docker.com/u/madminertool/
[repo-madminer]: https://github.com/diana-hep/madminer
[web-madgraph-main-page]: https://launchpad.net/mg5amcnlo
[web-madgraph-systematics]: https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/Systematics
