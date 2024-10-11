# HPS/ALP analysis code

This repo contains python3 code related to the second search for Higgs-portal scalars (HPS) decaying to electron positron pairs in MicroBooNE, with functionality to reweight for Axion-Like Particles.
It can be used to make various plots relating to the analysis and calculate limits on the HPS mixing angle theta, or mass and ALP-fermion coupling c_{\phi}. 

In the order which they should be run the three scripts are: 
1) Load the pickle files.
2) Calculate the expected and observed limits.
3) Plot the limits compared to other existing limits.

## Usage

Various packages are required to run these notebooks, the .yml file for the environment I used is included in the repo. 
Use conda to install a new virtual environment from this .yml file. Note I am on linux, so there may be some differences for mac users.

The code so far REQUIRES the .pkl files for all of the samples to have already been created.
These .pkl files can be created using the sample .root files and a function written by Pawel Guzowski.

## Authors

Joseph Bateman, joseph.bateman@manchester.ac.uk
David Marsden, david.marsden@manchester.ac.uk
Pawel Guzowski, pawel.guzowski@manchester.ac.uk
Some of the code was based on code written by Aditya Bhanderi for his thesis analysis.  
