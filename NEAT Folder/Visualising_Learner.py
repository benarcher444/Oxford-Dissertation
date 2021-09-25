# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 16:03:37 2021

@author: Ben Archer

This program visualises the swimmer over 30 seconds. Place the swimmers you would like to simulate
in the folders array. You can then alter the rate and speed of the visualisation.
"""

import pickle
import pygame
import sys
import time

folders = [
         "3 - Link 𝛿 = 1.5"
        ,"4 - Link"
        ,"5 - Link"
        ,"6 - Link"
        ,"10 - Link"
         ]

rate = 1000
runtime = 30
speed = 2

width = 600
height = 600
black = (0,0,0)
white = (255, 255, 255)
red   = (200, 0, 0)
blue = (0, 0, 200)

class Swimmer():
    def __init__(self, folder, number, total_number):
        self.y_displacement = height * (number + 1)/(total_number + 1)
        data = pickle.load(open(folder + '//data' + str(rate) + '.dat', 'rb'))
        pickle.dump(data, open(folder + '//data' + str(rate) + '.dat', 'wb'))
        self.alphas = data[3]
        self.coords_list = data[4]
        self.N = int(len(self.coords_list[0])/2 - 1)
        
    def draw_swimmer(self, screen, clock, tick_count, done):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        coords = self.coords_list[tick_count]
        
        pygame.draw.line(screen, black, (coords[0], coords[self.N + 1] + self.y_displacement) , (coords[1], coords[self.N + 2] + self.y_displacement))
        
        for i in range(1, self.N):
            pygame.draw.line(screen, black, (coords[i], coords[self.N + 1 + i] + self.y_displacement) , (coords[i + 1], coords[self.N + i + 2] + self.y_displacement))
            pygame.draw.circle(screen, red, (int(coords[i]), int(coords[self.N + 1 + i] + self.y_displacement)), 2)
        
        return done

def run():
    number_of_swimmers = len(folders)
    swimmers = []
    
    for i in range(number_of_swimmers):
        swimmers.append(Swimmer(folders[i], i, number_of_swimmers))
    
    done = False
    
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Micro Swimmers")
    
    tick_count = 0
    
    while not done:
        screen.fill(white)
        
        for swimmer in swimmers:
            done = swimmer.draw_swimmer(screen, clock, tick_count, done)
            if done == True:
                pygame.quit()
                sys.exit()
        
        pygame.display.flip()
        
        tick_count += speed
        if tick_count % (runtime*rate/10) == 0:
            print("Percentage Done:", tick_count/(runtime*rate))
        if tick_count > runtime*rate:
            done = True 
        clock.tick(rate)
    
    pygame.quit()
    
if __name__ == '__main__':
    start_time = time.time()
    run()
    print("Time Taken:", time.time() - start_time)