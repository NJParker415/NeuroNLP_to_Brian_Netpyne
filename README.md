# About the Project

This is a pipeline capable of transferring [NeuroNLP Medulla](https://medulla.neuronlp.fruitflybrain.org/) queries into the [NetPyNE](http://www.netpyne.org/) and [Brian2](https://brian2.readthedocs.io/en/stable/) neural simulators, used for a research project comparing the capabilities of said simulators. The package also contains support for transferring generated NetPyNE models to the [NeuroML2](https://docs.neuroml.org/Userdocs/NeuroMLv2.html) format, which in turn supports model transfer to various other neural simulators.

# Dependencies

* [brian2](https://brian2.readthedocs.io/en/stable/)
* [brian2tools](https://brian2tools.readthedocs.io/en/stable/)
* [cython](https://cython.org/)
* [flybrainlab](https://flybrainlab.fruitflybrain.org/)
* [graphviz](https://graphviz.org/)
* [gsl](https://www.gnu.org/software/gsl/)
* [install-jdk](https://pypi.org/project/install-jdk/)
* [lxml](https://lxml.de/)
* [netpyne](http://www.netpyne.org/)
* [pyneuroml](https://docs.neuroml.org/Userdocs/Software/pyNeuroML.html)

# Installation

The files in the ```package``` folder should be manually inserted into a project. Installation with pip or conda is not supported at this time.

# Usage

Programs using the files in this project should be run in a flybrainlab Medulla client, the process of setting up which is found [here](https://github.com/FlyBrainLab/Tutorials).

Instructions for defining custom components can be found in the comments of ```nlpToNetpyne.py``` and ```nlptoBrian2.py``` for NetPyNE and Brian2, respectively. These definitions broadly reflect the process by which components are defined in the original programs.

# Testing

```package_testing.ipnyb``` is a jupyter notebook that, when run in a flybrainlab Medulla client, queries increasing numbers of motor columns and measures each simulators performance in terms of runtime and computational load. By default, all cells are defined as Hodgkin-Huxley neurons and all synapses are defined by simple excitatory mechanisms. Individual cells and synapses can be custom-defined, though that functionality is not used here.
