# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 17:01:13 2021

@author: Ben Archer

This code does the actual training of the microswimmers. This code takes a long time to fully run
but does also allow you to see each microswimmer being trained.
"""

from __future__ import print_function
import os
os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"
import neat
import visualize
import sys
import pickle
import numpy as np
from numpy import sin, cos
import random
import pygame
random.seed(42)

'''
Here we can change some of the hyperparameters for the NEAT algorithm. 
The majority of the hyperparameters are fixed in the config file and can be altered there.

The load in variable allows you to pick up from a previously saved generation checkpoint. When set
to False it will just start a new process. To load in a generation just set it equal to say 30
if you wanted to load neat-checkpoint-30.
'''

pop_size = 100
number_of_generations =  31
load_in = False
new_fitness = False
mutate_power = False
mutate_rate = False
num_hidden = 0
compatibility_disjoint_coefficient = 0.9
compatibility_weight_coefficient = 0.9

'''
Variables for the microswimmers themselves. Make sure you set them to train the right type of swimmer.
The program will automatically create the right amount of inputs and outputs in the network.
'''

N = 3
L = np.array([20 for i in range(N)])
rate = 100
speed = 1
delta = 1.5/speed
max_angle = delta/2
runtime = 30

xi = 0.38*10**(-3)
eta = 1.89*xi

width = 600
height = 600
black = (0,0,0)
white = (255, 255, 255)
red   = (200, 0, 0)
blue = (0, 0, 200)

C = np.identity(N + 2)
C[3:N+2, 2:N + 1] += -np.identity(N - 1)
Cinv = np.linalg.inv(C).astype(int)

Q = np.zeros((N*3, N + 2), dtype = float)
Q[:N, 0] = np.ones(N)
Q[N:2*N, 1] = np.ones(N)
Q[2*N: , 2:] = np.identity(N)

P = np.zeros((3, 3*N), dtype = float)
coords = np.zeros((2, N + 1), dtype = float)

#This function simulates each generation, then records each indiduals fitness

def eval_genomes(genomes, config):
    done = False
    
    screen = pygame.display.set_mode((width,height))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Micro Swimmers")
    
    #creates a swimmer class object for each in the population
    
    swimmers = []
    
    i = 0
    for genome_id, genome in genomes:
        swimmers.append(Swimmer(i, pop_size, genome, config))
        i += 1
    
    tick_count = 0
    
    while not done:
    
        screen.fill(white)
        
        #iterates and draws each swimmer.
        
        for swimmer in swimmers:
            done = swimmer.iterate(screen, clock, done)
            if done == True:
                pygame.quit()
                sys.exit()
        
        pygame.display.flip()
        
        #reports progress of each generation
        
        tick_count += 1
        if tick_count % (runtime*rate/10) == 0:
            print("Percentage Done:", int(tick_count/(runtime*rate)*100))
        if tick_count > runtime*rate:
            done = True 
        clock.tick(rate)
    
    #records the fitness of each individual
    
    i = 0
    for genome_id, genome in genomes:
        reward = swimmers[i].finish() 
        genome.fitness = reward
        i += 1
    
    pygame.quit()

def run(config_file):
    # Load configuration.
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    #changes the congif file to match our preset conditions at the top of this file
    
    config.pop_size = pop_size
    config.genome_config.num_inputs = 3*N - 2
    config.genome_config.num_outputs = N - 1
    config.genome_config.input_keys = [-i for i in range(1, config.genome_config.num_inputs+1)]
    config.genome_config.output_keys = [i for i in range(config.genome_config.num_outputs)]
    config.genome_config.compatibility_disjoint_coefficient = compatibility_disjoint_coefficient
    config.genome_config.compatibility_weight_coefficient = compatibility_weight_coefficient
    config.genome_config.num_hidden = num_hidden
    
    if load_in != False:
        if type(load_in) == int:
            p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-" + str(load_in))
            old_stats = pickle.load(open("stats.dat", "rb"))
        else:
            p = neat.Checkpointer.restore_checkpoint(str(load_in))
            old_stats = pickle.load(open(str(load_in) + "-stats.dat", "rb"))
        if new_fitness != False:
            p.config.fitness_threshold = new_fitness
    else:
        p = neat.Population(config)
        
    if mutate_power != False:
        p.config.bias_mutate_power = mutate_power
        p.config.weight_mutate_power = mutate_power
    if mutate_rate != False:
        p.config.bias_mutate_rate = mutate_rate
        p.config.weight_mutate_rate = mutate_rate
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    
    #runs the neat algorithm
    
    winner = p.run(eval_genomes, number_of_generations)
    
    if load_in != False:
            old_stats.most_fit_genomes.extend(stats.most_fit_genomes)
            old_stats.generation_statistics.extend(stats.generation_statistics)
            visualize.plot_stats(old_stats, ylog=False, view=True)
            visualize.plot_species(old_stats, view=True)
            stats = old_stats
    else:
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)
        
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    visualize.draw_net(config, winner, True)
    
    return winner, stats

#The swimmer class

class Swimmer:
    #Initialises each swimmer.
    def __init__(self, number, total_number, genome, config):
        self.x1 = 10   
        self.y1 = height * (number + 1)/(total_number + 1)
        
        self.theta = np.zeros(N)
        self.alpha = np.zeros(N - 1)
        self.alphadot = np.zeros(N-1)
        
        self.x1_start = self.calculate_dist()
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.last_side = np.zeros(N - 1) - 1
    
    #reports the fitness
    def finish(self):
        return self.calculate_dist() - self.x1_start
    
    #Moves the swimmer one timestep    
    def iterate(self, screen, clock, done):
        coords = self.calculate_coords(self.x1, self.y1, self.theta)
        done = self.draw_swimmer(screen, clock, coords, done)
        
        inputs = np.concatenate([[self.theta[0]], self.alpha, self.alphadot, self.last_side])
        out = np.array(self.net.activate(inputs))
        
        self.set_rates(out)
        self.RK4()
        
        self.alpha = self.alpha + self.alphadot/rate
        self.calculate_thetas()
        
        return done
    
    #fixes the swimmer if it tries to overextend and sets the lambda values
    def set_rates(self, out):
        self.alphadot = np.array(2*out - 1)
        alpha = self.alpha + self.alphadot/rate
        self.last_side = [-self.last_side[i] if abs(alpha[i]) >= max_angle
                          else self.last_side[i] for i in range(N - 1)]
        self.alphadot = np.array([0 if abs(alpha[i]) >= max_angle
                                 else self.alphadot[i] for i in range(N - 1)])
    
    #calculates the steps using RK4
    def RK4(self):
        h = 1/(rate)
        xk1, yk1, tk1 = self.gradient(self.x1, self.y1, self.theta[0])
        xk2, yk2, tk2 = self.gradient(self.x1 + h*xk1/2, self.y1 + h*yk1/2, self.theta[0] + h*tk1/2)
        xk3, yk3, tk3 = self.gradient(self.x1 + h*xk2/2, self.y1 + h*yk2/2, self.theta[0] + h*tk2/2)
        xk4, yk4, tk4 = self.gradient(self.x1 + h*xk3, self.y1 + h*yk3, self.theta[0] + h*tk3)
        self.x1 = self.RK4Addition(self.x1, xk1, xk2, xk3, xk4, h)
        self.y1 = self.RK4Addition(self.y1, yk1, yk2, yk3, yk4, h)
        self.theta[0] = self.RK4Addition(self.theta[0], tk1, tk2, tk3, tk4, h)

    def RK4Addition(self, x, k1, k2, k3, k4, h):
        return x + h/6*(k1 + 2*k2 + 2*k3 + k4)
    
    #Calculates the derivates of x1, y1 and theta1

    def gradient(self, x1, y1, theta0):
        theta = np.append([theta0], self.theta[1:])
        coords = self.calculate_coords(x1, y1, theta)
        M = self.construct_matrices(theta, coords)
        Nmat = M @ Cinv
        A = Nmat[:,:3]
        B = Nmat[:,3:]
        Ainv = np.linalg.inv(A)
        return - Ainv @ B @ self.alphadot
    
    #Constructs the M matrix

    def construct_matrices(self, theta, coords):
        for i in range(0, N - 1):
            Q[i + 1, 2:(3 + i)] = - L[:(i+1)]*sin(theta[:(i+1)])
            Q[i + 1 + N, 2:(3 + i)] = L[:(i+1)]*cos(theta[:(i+1)])
   
        #Q[:N, 2:] = np.tril(- L*sin(theta), -1)
        #Q[N:(2*N), 2:] = np.tril(L*cos(theta), -1)
        
        #Q = np.block([[Q1, Q2, A],[Q2, Q1, B],[Q2, Q2, Q3]])
        
        change_x = coords[0, :N] - coords[0, 0]
        change_y = coords[1, :N] - coords[1, 0]

        P[0, :N] = L * (- xi*cos(theta)*cos(theta) - eta*sin(theta)*sin(theta))
        P[0, N:2*N] = L * sin(theta)*cos(theta) * (-xi + eta)
        P[0, 2*N:] = eta * L**2/2 * sin(theta)
 
        P[1, :N] = L * sin(theta)*cos(theta) * (-xi + eta)
        P[1, N:2*N] = L * (- xi*sin(theta)**2 - eta*cos(theta)**2)
        P[1, 2*N:] = - eta * L**2/2 * cos(theta)
  
        P[2, :N] = - change_x * L * sin(theta)*cos(theta) * (xi - eta) - change_y * L * (- xi * cos(theta)**2 - eta * sin(theta)**2) + eta * L**2/2 * sin(theta)
        P[2, N:2*N] = - change_x * L * (xi*sin(theta)**2 + eta*cos(theta)**2) - change_y * L * sin(theta)*cos(theta) * (- xi + eta) - eta * L**2/2 * cos(theta)
        P[2, 2*N:] = - change_x * eta * L**2/2 * cos(theta) - change_y * eta * L**2/2 * sin(theta) - eta * L**3/3

        return P @ Q
    
    # Calculates the coordinates of the swimmer at this point in time

    def calculate_coords(self, x, y, theta):
        coords[0, 0] = x
        coords[1, 0] = y
        coords[0, 1:] = x + np.cumsum(L * cos(theta))
        coords[1, 1:] = y + np.cumsum(L * sin(theta))
        return coords
    
    #Calculates the theta values from the alpha values

    def calculate_thetas(self):
        self.alpha[0] += self.theta[0]
        self.theta[1:] = np.cumsum(self.alpha)
        self.alpha[0] += - self.theta[0]    
        
    #Calculates the x coordinate of the end of the swimmer.
        
    def calculate_dist(self):
        return self.x1 + np.sum(L * cos(self.theta))
    
    #draws the swimmer to the screen

    def draw_swimmer(self, screen, clock, coords, done):                
        
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
        
        pygame.draw.line(screen, black, (coords[0, 0], coords[1, 0]) , (coords[0, 1], coords[1, 1]))
        
        for i in range(1, N):
            pygame.draw.line(screen, black, (coords[0, i], coords[1, i]) , (coords[0, i + 1], coords[1, i + 1]))
            pygame.draw.circle(screen, red, (int(coords[0, i]), int(coords[1, i])), 2)
        
        return done

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    winner, stats = run(config_path)
    pickle.dump(winner, open('winner.dat', 'wb'))
    pickle.dump(stats, open('stats.dat', 'wb'))
    print(stats)
