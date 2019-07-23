# Trouble shooting

If you are having issues with MadMiner, please go through the following check list:


## Event generation crashing

- Is MadGraph correctly installed? Can you generate events with MadGraph on its own, including the reweighting option?
- If you are using Pythia and Delphes: Are their installations working? Can you run MadGraph with Pythia, and can you run Delphes on the resulting HepMC
sample?
- If you are using PDF or scale uncertainties: Is LHAPDF installed with Python support?

## Key errors when reading LHE files

- Do LHE files contain multiple weights, one for each benchmark, for each event?

## Zero events after reading LHE or Delphes file

- Are there typos in the definitions of required observables, cuts, or efficiencies? If an observable, cut, or
efficiency causes all events to be discarded, DEBUG-level logging output should help you narrow down
the source.

## Neural network output does not make sense

- Start simple: one or two hidden layers are often enough for a start.
- Does the loss go down during training? If not, try changing the learning rate.
- Are the loss on the training and validation sample very different? This is the trademark sign of overtraining. Try
a simpler network architecture, more data, or early stopping.
