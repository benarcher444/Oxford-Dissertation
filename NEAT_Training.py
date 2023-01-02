# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 17:01:13 2021

@author: Ben Archer

This code trains the microswimmers through the NEAT algorithm. It starts either form a set checkpoint or from scratch.
It outputs the winner, the stats and plots of the training process to a folder within the training output folder.

Set the run_name variable to set the name of the folder to output to in the training output folder.
Set the pop_size variable to the size of the population each generation in the NEAT algorithm.
Set the number_of_generations to the max number of generations you want.
Set the runtime to the length of time each generation should have to move.
Set the rate to the timestep you want to train the learners at (Recommended 100).
Load in allows you to continue the training from a previous specific checkpoint.
Set Load in to an integer if you want to load from a specific checkpoint. Alternatively set it to a string if the
checkpoint name is a different string. If you do not want to load in, leave this False.
For new_fitness, mutate_power, mutate_rate set these to a float value instead of 0/False if you wish to change them.
Num hidden is the number of hidden nodes in the neural network policy when the training begins.

N is the number of links in the swimmers being trained
L is the length of each link
delta is the maxiumum angle that each link can stretch to (assuming that speed is 1)
"""

# Import required modules
import os
import sys
import neat
import visualize
import pickle
import pygame
import random

# Import configuration variables from run_config module
from run_config import white, height, width, neat_config, training_dir, checkpoint_dir

# Import the Learning Swimmer class for the training
from swimmer_classes import Learning_Swimmer

# Import functions to iterate the swimmer and calculate it's fitness
from helper_functions import iterate

os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"

# Remove this for new training
random.seed(1)

run_name = 'Test'
pop_size = 10
number_of_generations = 30
runtime = 15
rate = 100
load_in = False
new_fitness = False
mutate_power = False
mutate_rate = False
num_hidden = 0

N = 3
length = 20
delta = 1.5


def eval_genomes(genomes, config):
    """
    Calculates the fitness for each genome

    Parameters:
        genomes: a list of genome_ids and genomes
        config: a file specifying all of the necessary neat configurations
    """
    # Set up Pygame window
    done = False

    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Micro Swimmers")

    # Create each of the swimmers
    swimmers = []
    for i, (genome_id, genome) in enumerate(genomes):
        swimmers.append(Learning_Swimmer(N, length, delta, rate, i, pop_size, genome, config))

    tick_count = 0
    while not done:
        screen.fill(white)

        # Checks for quit on pygame window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Iterates and draws each swimmer.
        for swimmer in swimmers:
            iterate(swimmer, screen, tick_count)

        pygame.display.flip()

        # Reports progress of each generation
        tick_count += 1
        if tick_count % (runtime * rate / 10) == 0:
            print("Percentage Done:", int(tick_count / (runtime * rate) * 100))
        if tick_count > runtime * rate:
            done = True
        clock.tick(rate)

    print(genomes)
    # Records the fitness of each individual
    for genome, swimmer in zip(genomes, swimmers):
        genome[1].fitness = swimmer.calculate_fitness()

    pygame.quit()


def run():
    # Creates necesssary folders if they don't already exist
    if not os.path.exists(training_dir):
        os.mkdir(training_dir)

    dir_path = os.path.join(training_dir, run_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    checkpoint_dir_path = os.path.join(dir_path, checkpoint_dir)
    if not os.path.exists(checkpoint_dir_path):
        os.mkdir(checkpoint_dir_path)

    # Changes the config file to match our preset conditions at the top of this file

    neat_config.pop_size = pop_size
    neat_config.genome_config.num_inputs = 3 * N - 2
    neat_config.genome_config.num_outputs = N - 1
    neat_config.genome_config.input_keys = [-i for i in range(1, neat_config.genome_config.num_inputs + 1)]
    neat_config.genome_config.output_keys = [i for i in range(neat_config.genome_config.num_outputs)]
    neat_config.genome_config.num_hidden = num_hidden

    # Load in indicates if we want to continue our training from a previously saved checkpoint
    # We can either enter an integer to take it up from that checkpoint or a string indicating a manually entered checkpoint.
    # If we want to set a new fitness target we set new_fitness equal to it, otherwise it stays at 0

    checkpoint_path = os.path.join(checkpoint_dir_path, 'Checkpoint - ')
    stats_path = os.path.join(dir_path, 'stats.dat')
    winner_path = os.path.join(dir_path, 'winner.dat')
    net_path = os.path.join(dir_path, 'digraph.gv')
    fitness_path = os.path.join(dir_path, 'avg_fitness.svg')
    speciation_path = os.path.join(dir_path, 'speciation.svg')

    if load_in:
        if type(load_in) == int:
            p = neat.Checkpointer.restore_checkpoint(checkpoint_path + str(load_in))
            old_stats = pickle.load(open(stats_path, "rb"))
        else:
            p = neat.Checkpointer.restore_checkpoint(str(load_in))
            old_stats = pickle.load(open(str(load_in) + "-stats.dat", "rb"))

        if new_fitness:
            p.config.fitness_threshold = new_fitness

    else:
        p = neat.Population(neat_config)
        old_stats = None

    # Similar to above if we want to manually change the mutate power or mutate rate we can do it here.

    if mutate_power:
        p.config.bias_mutate_power = mutate_power
        p.config.weight_mutate_power = mutate_power

    if mutate_rate:
        p.config.bias_mutate_rate = mutate_rate
        p.config.weight_mutate_rate = mutate_rate

    # Adds reporters to log progress of the NEAT process.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, filename_prefix=checkpoint_path))

    # Runs the NEAT process. Takes the eval_genomes function and number_of_generations as parameters
    winner = p.run(eval_genomes, number_of_generations)

    # Ammends the stats file to include the old stats if it was loaded in.
    if load_in:
        old_stats.most_fit_genomes.extend(stats.most_fit_genomes)
        old_stats.generation_statistics.extend(stats.generation_statistics)
        stats = old_stats

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    visualize.draw_net(neat_config, winner, True, filename=net_path)
    visualize.plot_stats(stats, ylog=False, view=True, filename=fitness_path)
    visualize.plot_species(stats, view=True, filename=speciation_path)

    # Save the winner and stats file
    pickle.dump(winner, open(winner_path, 'wb'))
    pickle.dump(stats, open(stats_path, 'wb'))


if __name__ == '__main__':
    run()
