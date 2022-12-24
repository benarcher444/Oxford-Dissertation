import os
import pygame
import time
import sys
import pickle

from run_config import width, height, white, neat_config, training_dir
from swimmer_classes import Learning_Swimmer, Purcell_Swimmer, Test1_Swimmer, Test2_Swimmer
from env_functions import iterate_and_report, set_up_config, calculate_N, compile_swimmers
from visualize import plot_x_dist, plot_y_dist, plot_final_distances, plot_2d_displacements, plot_2d_alphas, plot_3d_alphas, plot_all_alphas, plot_positions

swimmer_creation = [{'type': 'Purcell', 'N': 3, 'length': 20, 'delta': 1.5, 'label': '3-Link Purcell', 'compile': '3 Link Purcell'},
                    {'type': 'Purcell', 'N': 4, 'length': 20, 'delta': 1.5, 'label': '4-Link Purcell', 'compile': '4 Link Purcell'},
                    {'type': 'Purcell', 'N': 5, 'length': 20, 'delta': 1.5, 'label': '5-Link Purcell', 'compile': '5 Link Purcell'},
                    {'type': 'Test1', 'N': 3, 'length': 20, 'delta': 1.5, 'label': '3-Link Test1', 'compile': '3 Link Wave'},
                    {'type': 'Test2', 'N': 3, 'length': 20, 'delta': 1.5, 'label': '3-Link Test2', 'compile': '3 Link Scallop'},
                    {'type': 'Learner', 'path': 'Test', 'length': 20, 'delta': 1.5, 'label': 'Learner', 'compile': 'Compiled Test'},
                    {'type': 'Learner', 'path': '10 - Link', 'length': 20, 'delta': 1.5, 'label': '10 - Link Learner', 'compile': '10 - Link'}]

rate = 100
runtime = 30
number_of_overlays = 4
plot = False
compile = True


def run():
    done = False

    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    pygame.display.set_caption("Micro Swimmers")

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
            path = os.path.join(training_dir, swimmer['path'], 'winner.dat')
            print(f"Loading Learner from Path {path}")
            genome = pickle.load(open(path, 'rb'))
            N = calculate_N(genome)
            config = set_up_config(neat_config, N)
            swimmers.append(Learning_Swimmer(N, swimmer['length'], swimmer['delta'], rate, i, total_count, genome, config))

        else:
            raise Exception("Wrong Swimmer Type Entered")

    tick_count = 0
    while not done:
        screen.fill(white)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        for swimmer in swimmers:
            iterate_and_report(swimmer, screen, tick_count)

        pygame.display.flip()

        tick_count += 1
        if tick_count % (runtime * rate / 10) == 0:
            print("Percentage Done:", int(tick_count / (runtime * rate) * 100))
        if tick_count > rate * runtime:
            done = True

        clock.tick(rate)

    pygame.quit()

    total_ticks = tick_count + 1

    if plot:
        plot_x_dist(swimmers, swimmer_creation, total_ticks)
        plot_y_dist(swimmers, swimmer_creation, total_ticks)
        plot_final_distances(swimmers, swimmer_creation)
        plot_2d_displacements(swimmers, swimmer_creation)
        plot_2d_alphas(swimmers, swimmer_creation)
        plot_3d_alphas(swimmers, swimmer_creation)
        plot_all_alphas(swimmers, swimmer_creation, total_ticks)
        plot_positions(swimmers, number_of_overlays)

    if compile:
        compile_swimmers(swimmers, swimmer_creation, rate, runtime)


if __name__ == "__main__":
    start_time = time.time()
    run()
    print(f"Time Taken:  {time.time() - start_time}")
