# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 14:15:51 2021

@author: Ben Archer

This code allows visualisation of different types of swimmer in real time moving through their
viscous environment. With this code you can simulate and visualise both types of Test swimmer,
along with the N-link Purcell Swimmer. Furthermore, it can simulate any NEAT swimmer.

This program computes each step as it runs, it does not use precomputed data. This means that
while it is easy to experiment and vary conditions, it can be slow to run. At the end of this program,
if the compile field is not None it will save and compile all of this precomputed data to a folder
of your name choice - speficied in that field. You can then simulate and plot these swimmers without
lag using the comparing_compiled_swimmers program.

Furthermore, this program can produce plots of the data, however if you want the plots quickly and the
swimmer has already been compiled, use the comparing_compiled_swimmers program.

To input new swimmers to this function add a new dict to the swimmer_creation array. You will need to
enter certain fields:

type: Purcell/Test1/Test2/Learner
N: The number of links in the swimmer
length: The length of each link
delta: The max angle value of each alpha (assuming speed=1)
label: The label to be shown on plots
path: For learner swimmers only - specifies the folder name where the winner.dat file resides
compile: If left blank/not entered the swimmer will not be compiled. It is the name of the file that the
         compiled swimmer data will be outputted to.

Ensure that the rate and runtime are at the level that you'd like to simulate and compile at

If you would like to plot and compile at all, ensure that plot and compile are both set to True
"""
# Import required modules
import os
import pygame
import time
import sys
import pickle

# Import configuration variables from run_config module
from run_config import width, height, white, neat_config, training_dir

# Import the different Swimmer classes for the simulation
from swimmer_classes import Learning_Swimmer, Purcell_Swimmer, Test1_Swimmer, Test2_Swimmer

# Import functions from env_functions for the iteration, set_up and compilation of swimmers
from helper_functions import iterate_and_report, set_up_config, calculate_N, compile_swimmers

# Import plotting functions from visualize
from visualize import plot_x_dist, plot_y_dist, plot_final_distances, plot_2d_displacements, plot_2d_alphas, plot_3d_alphas,\
                      plot_all_alphas, plot_positions, plot_positions_spaced

# The swimmer creation array: enter dicts to add new swimmers to the simulation
swimmer_creation = [{'type': 'Purcell', 'N': 3, 'length': 20, 'delta': 1.5, 'label': '3-Link Purcell', 'compile': '3 Link Purcell'},
                    {'type': 'Purcell', 'N': 4, 'length': 20, 'delta': 1.5, 'label': '4-Link Purcell', 'compile': '4 Link Purcell'},
                    {'type': 'Purcell', 'N': 5, 'length': 20, 'delta': 1.5, 'label': '5-Link Purcell', 'compile': '5 Link Purcell'},
                    {'type': 'Test1', 'N': 3, 'length': 20, 'delta': 1.5, 'label': '3-Link Test1', 'compile': '3 Link Wave'},
                    {'type': 'Test2', 'N': 3, 'length': 20, 'delta': 1.5, 'label': '3-Link Test2', 'compile': '3 Link Scallop'},
                    {'type': 'Learner', 'path': 'Test', 'length': 20, 'delta': 1.5, 'label': 'Learner', 'compile': 'Compiled Test'},
                    {'type': 'Learner', 'path': '10 - Link', 'length': 20, 'delta': 1.5, 'label': '10 - Link Learner', 'compile': '10 - Link'}]

rate = 100
runtime = 30
plot = False
compile = True


def run():
    # Creates swimmers
    swimmers = list()
    total_count = len(swimmer_creation)
    for i, swimmer in enumerate(swimmer_creation):
        if swimmer['type'] == 'Purcell':
            swimmers.append(Purcell_Swimmer(swimmer['N'], swimmer['length'], swimmer['delta'], rate, i, total_count))

        elif swimmer['type'] == 'Test1':
            swimmers.append(Test1_Swimmer(swimmer['N'], swimmer['length'], swimmer['delta'], rate, i, total_count))

        elif swimmer['type'] == 'Test2':
            swimmers.append(Test2_Swimmer(swimmer['N'], swimmer['length'], swimmer['delta'], rate, i, total_count))

        elif swimmer['type'] == 'Learner':
            # Loads genome and calculates N from the genome
            path = os.path.join(training_dir, swimmer['path'], 'winner.dat')
            print(f"Loading Learner from Path {path}")
            genome = pickle.load(open(path, 'rb'))
            N = calculate_N(genome)
            config = set_up_config(neat_config, N)
            swimmers.append(Learning_Swimmer(N, swimmer['length'], swimmer['delta'], rate, i, total_count, genome, config))

        else:
            raise Exception(f"Wrong Swimmer Type Entered: {swimmer['type']}")

    # Set up pygame screen
    done = False

    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    pygame.display.set_caption("Micro Swimmers")
    tick_count = 0

    while not done:
        screen.fill(white)

        # Check for pygame exit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Iterate swimmers
        for swimmer in swimmers:
            iterate_and_report(swimmer, screen, tick_count)

        pygame.display.flip()

        # Display updates on percentage done
        tick_count += 1
        if tick_count % (runtime * rate / 10) == 0:
            print("Percentage Done:", int(tick_count / (runtime * rate) * 100))
        if tick_count > rate * runtime:
            done = True

        clock.tick(rate)

    pygame.quit()

    total_ticks = tick_count + 1

    # Plot the swimmers
    if plot:
        plot_x_dist(swimmers, swimmer_creation, total_ticks)
        plot_y_dist(swimmers, swimmer_creation, total_ticks)
        plot_final_distances(swimmers, swimmer_creation)
        plot_2d_displacements(swimmers, swimmer_creation)
        plot_2d_alphas(swimmers, swimmer_creation)
        plot_3d_alphas(swimmers, swimmer_creation)
        plot_all_alphas(swimmers, swimmer_creation, total_ticks)
        plot_positions(swimmers)
        plot_positions_spaced(swimmers)

    # Compile the swimmers
    if compile:
        compile_swimmers(swimmers, swimmer_creation, rate, runtime)


if __name__ == "__main__":
    start_time = time.time()
    run()
    print(f"Time Taken:  {time.time() - start_time}")
