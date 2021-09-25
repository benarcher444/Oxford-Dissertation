# Oxford-Dissertation
A repository for the code used in my Oxford Dissertation.

In the NEAT folder we have 5 different main programs and 8 different trained swimmers. 
Each program is annotated and has a brief description at the beginning.

Each swimmer's network/policy is stored in the winner.dat file, unique in each folder.
The Checkpoint file is the last run NEAT generation, you can use this file to continue training each swimmer further if need be.
If you do keep training, ensure that you copy this checkpoint file and the stats file into the same directory as NEAT_Training.py

You will need to install:

pygame
math
numpy
matplotlib
os
neat
itertools
pickle
sys
time
random

All of these that are not already installed can be installed using pip. 
