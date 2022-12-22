import pygame
import time
import sys
import pickle

from run_config import width, height, white, rate, neat_config
from swimmer_classes import Learning_Swimmer, Purcell_Swimmer, Test1_Swimmer, Test2_Swimmer
from env_functions import report_position, iterate
from visualize import plot_x_dist, plot_y_dist, plot_final_distances, plot_2d_displacements, plot_2d_alphas, plot_all_alphas, plot_positions

swimmer_creation = [{'type': 'Purcell', 'N': 3, 'length': 20, 'delta': 1.5, 'label': '3-Link Purcell'},
                    {'type': 'Purcell', 'N': 4, 'length': 20, 'delta': 1.5, 'label': '4-Link Purcell'},
                    {'type': 'Purcell', 'N': 5, 'length': 20, 'delta': 1.5, 'label': '5-Link Purcell'},
                    {'type': 'Test1', 'N': 3, 'length': 20, 'delta': 1.5, 'label': '3-Link Test1'},
                    {'type': 'Test2', 'N': 3, 'length': 20, 'delta': 1.5, 'label': '3-Link Test2'}
                    {'type': 'Learner', 'path': 'winner.dat', 'length': 20, 'delta': 1.5, 'label': 'Learner'}]

runtime = 10
number_of_overlays = 4
plotting = True


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
            swimmers.append(Purcell_Swimmer(swimmer['N'], swimmer['length'], swimmer['delta'], i, total_count))

        elif swimmer['type'] == 'Test1':
            swimmers.append(Test1_Swimmer(swimmer['N'], swimmer['length'], swimmer['delta'], i, total_count))

        elif swimmer['type'] == 'Test2':
            swimmers.append(Test2_Swimmer(swimmer['N'], swimmer['length'], swimmer['delta'], i, total_count))

        elif swimmer['type'] == 'Learner':
            genome = pickle.load(open(swimmer['path'], 'rb'))
            N = len(genome.nodes) + 1
            neat_config.genome_config.num_inputs = 3 * N - 2
            neat_config.genome_config.num_outputs = N - 1
            neat_config.genome_config.input_keys = [-i for i in range(1, neat_config.genome_config.num_inputs + 1)]
            neat_config.genome_config.output_keys = [i for i in range(neat_config.genome_config.num_outputs)]
            swimmers.append(Learning_Swimmer(N, swimmer['length'], swimmer['delta'], i, total_count, genome, neat_config))

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
            iterate(swimmer, screen, tick_count)
            print(swimmer.x1, swimmer.y1, swimmer.theta[0])

        pygame.display.flip()

        # prints progress

        tick_count += 1
        if tick_count % (runtime * rate / number_of_overlays) == 0:
            for swimmer in swimmers:
                report_position(swimmer)
        if tick_count % (runtime * rate / 10) == 0:
            print("Percentage Done:", int(tick_count / (runtime * rate) * 100))
        if tick_count > rate * runtime:
            done = True

        clock.tick(rate)

    pygame.quit()

    total_ticks = tick_count + 1

    if plotting:
        plot_x_dist(swimmers, swimmer_creation, total_ticks)
        plot_y_dist(swimmers, swimmer_creation, total_ticks)
        plot_final_distances(swimmers, swimmer_creation)
        plot_2d_displacements(swimmers, swimmer_creation)
        plot_2d_alphas(swimmers, swimmer_creation)
        plot_all_alphas(swimmers, swimmer_creation, total_ticks)
        plot_positions(swimmers)


if __name__ == "__main__":
    time_start = time.time()
    run()
    print("Time Taken:", time.time() - time_start)
