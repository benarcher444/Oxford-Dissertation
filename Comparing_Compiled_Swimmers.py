# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 16:03:37 2021

@author: Ben Archer

This program visualises the swimmer over 30 seconds. Place the folders or compiled swimmers
you would like to simulate in the folders array. You can then alter the rate and speed of the visualisation.
"""

import pickle
import pygame
import sys
import time
import os

from run_config import width, height, compiled_dir, white
from env_functions import draw_compiled_swimmer
from visualize import plot_x_dist, plot_y_dist, plot_final_distances, plot_2d_displacements, plot_2d_alphas,\
                      plot_all_alphas, plot_positions, plot_positions_spaced, plot_3d_alphas

folders = ["3 Link Purcell",
           "4 Link Purcell",
           "3 Link Scallop",
           "10 - Link"]

rate = 100
runtime = 30
play_speed = 2

plot = True
number_of_overlays = 10
draw = True


class Compiled_Swimmer:
    def __init__(self, data, number, total_number):
        self.x_dist = data['x_dist']
        self.y_dist = data['y_dist']
        self.thetas = data['thetas']
        self.alphas = data['alphas']
        self.coords_list = data['coords_list']
        self.dot_colour = data['dot_colour']
        self.N = self.coords_list[0].shape[1] - 1
        self.rate = rate

        y_displacement = height * (number + 1)/(total_number + 1)

        for i, coords in enumerate(self.coords_list):
            coords[1, :] = coords[1, :] + y_displacement


def run():
    done = False

    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    pygame.display.set_caption("Micro Swimmers")

    # Creates swimmers
    swimmers = list()
    labels = list()
    total_count = len(folders)
    for i, folder in enumerate(folders):

        file_name = f"compiled-rate-{rate}-runtime-{runtime}.dat"
        folder_path = os.path.join(compiled_dir, folder)
        if not os.path.exists(folder_path):
            raise Exception(f"No folder with name {folder} found at {folder_path}")
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            raise Exception(f"No compiled run with rate-{rate} and runtime-{runtime} found in folder {folder_path}\nNo File: {file_path}")

        data = pickle.load(open(file_path, 'rb'))

        swimmers.append(Compiled_Swimmer(data, i, total_count))
        labels.append({'label': data['label']})

    if plot:
        total_ticks = rate * runtime + 2
        plot_x_dist(swimmers, labels, total_ticks)
        plot_y_dist(swimmers, labels, total_ticks)
        plot_final_distances(swimmers, labels)
        plot_2d_displacements(swimmers, labels)
        plot_2d_alphas(swimmers, labels)
        plot_3d_alphas(swimmers, labels)
        plot_all_alphas(swimmers, labels, total_ticks)
        plot_positions(swimmers, number_of_overlays)
        plot_positions_spaced(swimmers, number_of_overlays)

    if draw:
        tick_count = 0
        while not done:
            screen.fill(white)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            for swimmer in swimmers:
                draw_compiled_swimmer(swimmer, screen, tick_count*play_speed)

            pygame.display.flip()

            tick_count += 1
            if tick_count * play_speed % (runtime * rate / 10) == 0:
                print("Percentage Done:", int(tick_count * play_speed / (runtime * rate) * 100))
            if tick_count * play_speed > rate * runtime:
                done = True

            clock.tick(rate)

        pygame.quit()


if __name__ == '__main__':
    start_time = time.time()
    run()
    print(f"Time Taken:  {time.time() - start_time}")
