# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 17:01:13 2021

@author: Ben Archer

This code does the actual training of the micro-swimmers. This code takes a long time to fully run
but does also allow you to see each micro-swimmer being trained.
"""

import os
import sys
import neat
import visualize
import pickle
import pygame
# import random

from run_config import rate, white, height, width, neat_config, training_path, checkpoint_dir
from swimmer_classes import Learning_Swimmer
from env_functions import calculate_fitness, iterate

os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"

# random.seed(1)

run_name = 'Test'
pop_size = 10
number_of_generations = 2
runtime = 15
load_in = False
new_fitness = False
mutate_power = False
mutate_rate = False
num_hidden = 0
compatibility_disjoint_coefficient = 0.9
compatibility_weight_coefficient = 0.7

N = 3
length = 20
delta = 1.5


def eval_genomes(genomes, config):
    done = False

    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Micro Swimmers")

    swimmers = []

    i = 0
    for genome_id, genome in genomes:
        swimmers.append(Learning_Swimmer(N, length, delta, i, pop_size, genome, config))
        i += 1

    tick_count = 0
    while not done:
        screen.fill(white)

        # iterates and draws each swimmer.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        for swimmer in swimmers:
            iterate(swimmer, screen, tick_count)

        pygame.display.flip()

        # reports progress of each generation
        tick_count += 1
        if tick_count % (runtime * rate / 10) == 0:
            print("Percentage Done:", int(tick_count / (runtime * rate) * 100))
        if tick_count > runtime * rate:
            done = True
        clock.tick(rate)

    print(genomes)
    # records the fitness of each individual
    for genome, swimmer in zip(genomes, swimmers):
        genome[1].fitness = calculate_fitness(swimmer)

    pygame.quit()


def run():
    if not os.path.exists(training_path):
        os.mkdir(training_path)

    dir_path = os.path.join(training_path, run_name)
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
    neat_config.genome_config.compatibility_disjoint_coefficient = compatibility_disjoint_coefficient
    neat_config.genome_config.compatibility_weight_coefficient = compatibility_weight_coefficient
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

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, filename_prefix=checkpoint_path))

    winner = p.run(eval_genomes, number_of_generations)

    if load_in:
        old_stats.most_fit_genomes.extend(stats.most_fit_genomes)
        old_stats.generation_statistics.extend(stats.generation_statistics)
        stats = old_stats

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    visualize.draw_net(neat_config, winner, True, filename=net_path)
    visualize.plot_stats(stats, ylog=False, view=True, filename=fitness_path)
    visualize.plot_species(stats, view=True, filename=speciation_path)

    pickle.dump(winner, open(winner_path, 'wb'))
    pickle.dump(stats, open(stats_path, 'wb'))


if __name__ == '__main__':
    run()

