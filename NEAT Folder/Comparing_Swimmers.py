# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 14:15:51 2021

@author: Ben Archer

This code allows visualisation of different types of swimmer in real time moving through their
viscous environment. With this code you can simulate and visualise both types of Test swimmer, 
along with the N-link Purcell Swimmer. Furthermore, it can simulate any NEAT swimmer.

This program computes each step as it runs, it does not use precomputed data. This means that 
while it is easy to experiment and vary conditions, it can be slow to run. If you are wanting to
visualise the NEAT swimmers, I would recommend using Visualising_Learner.py instead as this uses 
the precompiled data set and so will run without lag.

Furthermore, this program can produce plots of the data, however if you want to plot NEAT swimmer
results, I would again recommend using the precompiled program PLotting_Learner.py.
"""

import pygame
from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt
import os
import neat
import itertools
import pickle
import sys
import time

'''
As stated, this program is very malleable and we can easily change a lot of the conditions for each
swimmer. Here we have the diff_swimmers array. Each line produces a new swimmer in the visualisation.
Each row consists of a seperate list with 6 entries. 

1) The type of swimmer: Purcell, Test, Test2, or Learning
The first three are shown with red joints, but learning types are shown with blue.

2) The number of links if it is any of the first three types of swimmer
If it is a learning type, instead put the route to the winner file.

3) The length of each link in the swimmer
4) The value of delta/the maximum angle excursion
5) The name to appear in the legend when plotted
6) The colour of the line to appear in the legend when plotted 
'''

diff_swimmers = [["Purcell", 3, 20, 1.5, "Purcell Swimmer", "Red"]     
                ,["Test", 3, 20, 1.5, "Test: Wave", "BLue"]
                ,["Test2", 3, 20, 1.5, "Test: Scallop", "Green"]
                ,["Learning", "3 - Link 𝛿 = 1.5//winner.dat", 20, 1.5, "3-Link", "Black"]
                ,["Learning", "4 - Link//winner.dat", 20, 1.5, "3-Link", "Black"]
                ,["Learning", "5 - Link//winner.dat", 20, 1.5, "3-Link", "Black"]
                ,["Learning", "6 - Link//winner.dat", 20, 1.5, "3-Link", "Black"]
                ,["Learning", "10 - Link//winner.dat", 20, 1.5, "3-Link", "Black"]
                ]

'''
Here we have further alterable hyper-parameters

Speed: The speed the simulation runs at
Rate: 1/time-step or the number of frames per second. At high rates expect lag.
Runtime: The number of seconds to run the simulation for
'''

speed = 1
rate = 100
runtime = 1

'''
Below you can choose which different plots you want from a simulation.
Number of shots defines how many different images you want overlayed for the placements plot.
'''

x_displacement = True
distances = True
y_displacement = True
displacements_2D = True
placements = True
number_of_shots = 1
alphas = True
alphas_2D = True

xi = 0.38*10**(-3)
eta = 1.89 * xi

width = 600
height = 600
black = (0,0,0)
white = (255, 255, 255)
red   = (200, 0, 0)
blue = (0, 0, 200)

plot_colours1 = True
plot_colours2 = False
colors = ["Red", "Blue", "Green", "Purple", "Yellow"]

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

class Swimmer:
    def __init__(self, info, number, total_number, config = "None"):
        
        #initialises the conditions for each different kind of swimmer.
        
        self.mode = info[0]
        settings = info[1]
        link_length = info[2]
        self.delta = info[3]/speed
        self.max_angle = self.delta/2
    
        if self.mode == "Purcell":
            self.N = settings
            self.alpha = np.zeros(self.N - 1) - self.delta*speed/2
            self.theta = np.zeros(self.N)
            if self.N % 2 == 0:
                self.theta[0] = self.delta*speed/2 * (self.N//2 - 1) + self.delta*speed/4
            else:
                self.theta[0] = self.delta*speed/2 * (self.N//2)
            self.theta = calculate_thetas(self.theta, self.alpha, self.N)
            
        elif self.mode == "Test":
            self.N = settings
            self.theta = np.append(np.zeros(self.N-1),[-self.delta*speed/2])
            
        elif self.mode == "Test2":
            self.N = settings
            self.theta = np.zeros(self.N)
        
        elif self.mode == "Learning":
            genome = pickle.load(open(settings,'rb'))
            pickle.dump(genome, open(settings, 'wb'))
            
            self.N = int(calculate_N(genome) + 1)
            self.theta = np.zeros(self.N)
            
            config.genome_config.num_inputs = 3*self.N - 2
            config.genome_config.num_outputs = self.N - 1
            config.genome_config.input_keys = [-i for i in range(1, config.genome_config.num_inputs + 1)]
            config.genome_config.output_keys = [i for i in range(config.genome_config.num_outputs)]
            
            self.net = neat.nn.FeedForwardNetwork.create(genome, config)
            self.last_side = np.zeros(self.N - 1) - 1
            
        else:
            print("Incorrect Type of Swimmer Entered.")
            pygame.quit()
            sys.exit()
        
        self.L = np.array([link_length for i in range(self.N)])
        self.Cinv = calculate_Cinv(self.N)
        
        self.x1 = 10   
        self.y1 = height * (number + 1)/(total_number + 1)

        self.alphadot = np.zeros(self.N - 1)
        self.alpha = self.theta[1:self.N] - self.theta[:(self.N - 1)]
        
        self.x1_start, self.y1_start = calculate_dist(calculate_coords(self.x1, self.y1, self.theta, self.N, self.L), self.N)
        
        self.dist = [0]
        self.ydist = [0]
        self.alphas = np.array([self.alpha])
        
    def iterate(self, screen, clock, done, tick_count):
        coords = calculate_coords(self.x1, self.y1, self.theta, self.N, self.L)
        done = draw_swimmer(screen, clock, coords, done, self.N, self.mode)
        
        #Calculates new alpha derivative values.
        
        if self.mode == "Purcell":
            self.alphadot = Purcell(self.delta, tick_count, self.N)
            
        elif self.mode == "Test":
            self.alphadot = Test(self.delta, tick_count, self.N)
            
        elif self.mode == "Test2":
            self.alphadot = Test2(self.delta, tick_count, self.N)
        
        elif self.mode == "Learning":
            inputs = [[self.theta[0]], self.alpha, self.alphadot, self.last_side]
            flat_inputs = list(itertools.chain(*inputs))
            out = np.array(self.net.activate(flat_inputs))
            self.alphadot, self.last_side = np.array(set_rates(self.alpha, self.last_side, out, self.N, self.max_angle))
        
        self.x1, self.y1, self.theta[0] = RK4(self.x1, self.y1, self.theta[0], self.theta[1:], self.alphadot, self.N, self.L, self.Cinv)
        
        '''
        This is the Euler method version:
        
        xdot, ydot, thetadot = gradient(self.x1, self.y1, self.theta[0], self.theta[1:], self.alphadot, self.N, self.L, self.Cinv)
        
        self.x1 = self.x1 + xdot/rate
        self.y1 = self.y1 + ydot/rate
        self.theta[0] = self.theta[0] + thetadot/rate
        '''
        
        self.alpha = self.alpha + self.alphadot/rate
        self.theta = calculate_thetas(self.theta, self.alpha, self.N)
        
        x_dist, y_dist = calculate_dist(coords, self.N)
            
        self.dist.append(x_dist - self.x1_start)
        self.ydist.append(y_dist - self.y1_start)
        self.alphas = np.vstack((self.alphas, self.alpha))
        return done
    
    #reports the shape for the placements plot
    
    def report_shape(self):
        return calculate_coords(self.x1, self.y1, self.theta, self.N, self.L)
    
    #returns the output for plotting
    
    def finish(self):
        self.alphas = np.rollaxis(self.alphas, 1)
        return max(self.dist), self.dist, self.alphas, self.N, self.ydist
    
#Calculates N from the genome alone
    
def calculate_N(genome):
    out = 0
    for i in range(200):
        try:
            genome.nodes[i]
            out += 1
        except:
            break
    return out

#Runs the RK4 method
    
def RK4(x1, y1, theta0, theta1, alphadot, N, L, Cinv):
    h = 1/rate
    xk1, yk1, tk1 = gradient(x1, y1, theta0, theta1, alphadot, N, L, Cinv)
    xk2, yk2, tk2 = gradient(x1 + h*xk1/2, y1 + h*yk1/2, theta0 + h*tk1/2, theta1, alphadot, N, L, Cinv)
    xk3, yk3, tk3 = gradient(x1 + h*xk2/2, y1 + h*yk2/2, theta0 + h*tk2/2, theta1, alphadot, N, L, Cinv)
    xk4, yk4, tk4 = gradient(x1 + h*xk3, y1 + h*yk3, theta0 + h*tk3, theta1, alphadot, N, L, Cinv)
    
    xnew = RK4Addition(x1, xk1, xk2, xk3, xk4, h)
    ynew = RK4Addition(y1, yk1, yk2, yk3, yk4, h)
    tnew = RK4Addition(theta0, tk1, tk2, tk3, tk4, h)
    return xnew, ynew, tnew

def RK4Addition(x, k1, k2, k3, k4, h):
    return x + h/6*(k1 + 2*k2 + 2*k3 + k4)

#Calculates the derivatives
    
def gradient(x1, y1, theta0, theta1, alphadot, N, L, Cinv):
    theta = np.append([theta0], theta1)
    coords = calculate_coords(x1, y1, theta, N, L)
    M = construct_matrices(L, theta, coords, N)
    Nmat = M @ Cinv
    A = Nmat[:,:3]
    B = Nmat[:,3:]
    Ainv = np.linalg.inv(A)
    return - Ainv @ B @ alphadot

#Fixes rates if the swimmer tries to overextend, also changes lambda values

def set_rates(alpha, last_side, out, N, max_angle):
    alphadot = 2*out - 1
    alpha = alpha + alphadot/rate
    last_side = [-last_side[i] if abs(alpha[i]) >= max_angle else last_side[i] for i in range(N - 1)]
    alphadot = [0 if abs(alpha[i]) >= max_angle else alphadot[i] for i in range(N - 1)]
    return alphadot, last_side

#Calculates the M matrix

def construct_matrices(L, theta, coords, N):
    Q = np.zeros((N*3, N + 2), dtype = float)
    Q[:N, 0] = [1 for i in range(N)]
    Q[N:2*N, 1] = [1 for i in range(N)]
    Q[2*N: , 2:] = np.identity(N)
    for i in range(0, N - 1):
        Q[i + 1, 2:(3 + i)] = [- L[j]*sin(theta[j]) for j in range(i + 1)]
        Q[i + 1 + N, 2:(3 + i)] = [L[j]*cos(theta[j]) for j in range(i + 1)]
    
    P = np.zeros((3, 3*N), dtype = float)
    
    change_x = [coords[i][0] - coords[0][0] for i in range(N)]
    change_y = [coords[i][1] - coords[0][1] for i in range(N)]
    sincos = [sin(theta[i]) * cos(theta[i]) for i in range(N)]
    sinsin = [sin(theta[i])**2 for i in range(N)]
    coscos = [cos(theta[i])**2 for i in range(N)]
    
    P[0, :N] = [L[i] * (- xi*coscos[i] - eta*sinsin[i]) for i in range(N)]
    P[0, N:2*N] = [L[i] * sincos[i] * (-xi + eta) for i in range(N)]
    P[0, 2*N:] = [eta * L[i]**2/2 * sin(theta[i]) for i in range(N)]
    
    P[1, :N] = [L[i] * sincos[i] * (-xi + eta) for i in range(N)]
    P[1, N:2*N] = [L[i] * (- xi*sinsin[i] - eta*coscos[i]) for i in range(N)]
    P[1, 2*N:] = [- eta * L[i]**2/2 * cos(theta[i]) for i in range(N)]
    
    P[2, :N] = [- change_x[i] * L[i] * sincos[i] * (xi - eta) - change_y[i] * L[i] * (- xi * coscos[i] - eta * sinsin[i]) + eta * L[i]**2/2 * sin(theta[i]) for i in range(N)]
    P[2, N:2*N] = [- change_x[i] * L[i] * (xi * sinsin[i] + eta*coscos[i]) - change_y[i] * L[i] * sincos[i] * (- xi + eta) - eta * L[i]**2/2 * cos(theta[i]) for i in range(N)]
    P[2, 2*N:] = [- change_x[i] * eta * L[i]**2/2 * cos(theta[i]) - change_y[i] * eta * L[i]**2/2 * sin(theta[i]) - eta * L[i]**3/3 for i in range(N)]
    
    return P @ Q

#Calculates the Cinv matrix from N

def calculate_Cinv(N):
    C = np.identity(N + 2)
    C[3:N+2, 2:N + 1] += -np.identity(N - 1)
    return np.linalg.inv(C)

#Recalculates the swimmers theta values from the alphas

def calculate_thetas(theta, alpha, N):
    theta[1:] = np.cumsum(alpha) + theta[0]
    return theta

#Calculates coordinates of the full swimmer

def calculate_coords(x, y, theta, N, L):
    coords = [[x, y]]
    for i in range(N):
        x = coords[i][0] + L[i] * cos(theta[i])
        y = coords[i][1] + L[i] * sin(theta[i])
        coords.append(np.array([x, y]))
    return coords

#Calculates the midpoint distance

def calculate_dist(coords, N):
    if N % 2 == 0:
        return coords[int(N/2)][0], coords[int(N/2)][1]
    else:
        return (coords[int((N-1)/2)][0] + coords[int((N+1)/2)][0])/2, (coords[int((N-1)/2)][1] + coords[int((N+1)/2)][1])/2

#Draws the current swimmers position to the screen

def draw_swimmer(screen, clock, coords, done, N, mode):                
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                pygame
    
    pygame.draw.line(screen, black, (coords[0][0], coords[0][1]) , (coords[1][0], coords[1][1]))
    
    for i in range(1, N):
        pygame.draw.line(screen, black, (coords[i][0], coords[i][1]) , (coords[i+1][0], coords[i+1][1]))
        if mode == "Learning":
            pygame.draw.circle(screen, blue, (int(coords[i][0]), int(coords[i][1])), 2)
        else:
            pygame.draw.circle(screen, red, (int(coords[i][0]), int(coords[i][1])), 2)
    
    return done

#Controls alpha values for Purcell Swimmer

def Purcell(delta, tick_count, N):
    time_elapsed = tick_count/rate
    which = int((time_elapsed // delta) % (2 * (N - 1)))
    negative = 1 if which < (N - 1) else -1
    which = which % (N - 1)
    alphadot = np.zeros(N - 1)
    alphadot[which] = negative*speed
    return np.flip(alphadot)

#Controls alpha values for Wave Test
    
def Test(delta, tick_count, N):
    time_elapsed = tick_count/rate
    if time_elapsed % (delta*2) < delta:
        return np.append(np.zeros(N-2),[1])
    else:
        return np.append(np.zeros(N-2),[-1])

#Controls alpha values for Scallop Test

def Test2(delta, tick_count, N):
    time_elapsed = tick_count/rate
    if time_elapsed % (delta*1.5) < delta:
        return np.zeros(N-1) + 0.5
    else:
        return np.zeros(N-1) - 1
    
def main(config):    
    done = False
    
    screen = pygame.display.set_mode((width,height))
    clock = pygame.time.Clock()

    pygame.display.set_caption("Micro Swimmers")
    
    #Creates swimmers
    
    number_of_swimmers = len(diff_swimmers)
    swimmer_list = []
    positions = np.zeros((number_of_swimmers, number_of_shots + 1), dtype = list)
    
    for i in range(number_of_swimmers):
        swimmer_list.append(Swimmer(diff_swimmers[i], i, number_of_swimmers, config = config))
        positions[i][0] = swimmer_list[i].report_shape()
    
    tick_count = 0

    while not done:
        
        #iterates and draws the swimmers
        
        screen.fill(white)
        
        for i in range(number_of_swimmers):
            done = swimmer_list[i].iterate(screen, clock, done, tick_count)
        
        pygame.display.flip()
        
        #prints progress
        
        tick_count += 1
        if tick_count % (runtime*rate/number_of_shots) == 0:
            for i in range(number_of_swimmers):
                positions[i][int(tick_count/(runtime*rate/number_of_shots))] = swimmer_list[i].report_shape()
        if tick_count % (runtime*rate/10) == 0:
            print("Percentage Done:", int(tick_count/(runtime*rate)*100))
        if tick_count > rate*runtime:
            done = True
        
        clock.tick(rate)
        
    #Gets the results from the class objects
        
    out = []
    for i in range(number_of_swimmers):
        out.append(swimmer_list[i].finish())
    
    ###Plotting Section
    
    #Plotting x Displacements
    
    if x_displacement == True:
        for i in range(number_of_swimmers):
            total_ticks = len(out[i][1])
            time = np.linspace(0, total_ticks - 1, total_ticks)/rate
            if plot_colours1 == True:
                plt.plot(time, out[i][1], color = diff_swimmers[i][5], label = diff_swimmers[i][4])
            else:
                plt.plot(time, out[i][1], label = diff_swimmers[i][4])
            print(str(diff_swimmers[i][4]) + ": " + str(out[i][1][-1]))
    
        plt.legend()
        plt.title(r"$x$ Displacement of Different Swimmers over Time")
        plt.xlabel("Time (s)")
        plt.ylabel(r"$x$ Displacement (μm)")
        plt.show()
    
    if distances == True:
        finishes = []
        names = []
        for i in range(number_of_swimmers):
            finishes.append(out[i][1][-1])
            names.append(diff_swimmers[i][4])
        print(finishes)
        plt.plot(range(number_of_swimmers), finishes, marker='o')
        plt.title("Displacements of Different Swimmers")
        plt.xlabel("Swimmer")
        plt.ylabel(r"$x$ Displacement (μm)")
        plt.xticks(range(number_of_swimmers), names, rotation = 0)
        plt.show()
    
    #Plotting y Displacements
    
    if y_displacement == True:
        for i in range(number_of_swimmers):
            out.append(swimmer_list[i].finish())
            total_ticks = len(out[i][4])
            time = np.linspace(0, total_ticks - 1, total_ticks)/rate
            if plot_colours1 == True:
                plt.plot(time, out[i][4], color = diff_swimmers[i][5], label = diff_swimmers[i][4])
            else:
                plt.plot(time, out[i][4], label = diff_swimmers[i][4])
    
        plt.title(r"$y$ Displacement of Different Swimmers over Time")
        plt.xlabel("Time (s)")
        plt.ylabel(r"$y$ Displacement (μm)")
        plt.show()
    
    #Plotting 2D Displacements over time
    
    if displacements_2D == True:
        for i in range(number_of_swimmers):
            plt.plot(out[i][1], out[0][4], label = diff_swimmers[i][4])
            plt.xlabel(r"$x$ Displacement (μm)")
            plt.ylabel(r"$y$ Displacement (μm)")
            plt.show()
    
    #Plotting the Positions  
    
    if placements == True:
        for i in range(number_of_swimmers):
            col_change = [0, 0, 1]
            for j in range(len(positions[i])):
                xs = np.array([positions[i][j][k][0] for k in range(out[i][3] + 1)])
                ys = np.array([positions[i][j][k][1] for k in range(out[i][3] + 1)])
                plt.plot(xs, ys, color = tuple(col_change))
                col_change[0] += 1/(number_of_shots + 2)
                col_change[2] += -1/(number_of_shots + 2)
            plt.tick_params(axis = 'both', which = 'both', bottom = False, top = False, labelbottom = False, right = False, left = False, labelleft = False) # labels along the bottom edge are off
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
            plt.gca().set_aspect("equal")
            plt.show()

    #Plotting all Alphas
    
    if alphas == True:
        for i in range(number_of_swimmers):
            total_ticks = len(out[i][2][0])
            time = np.linspace(0, total_ticks - 1, total_ticks)/rate
            for j in range(out[i][3] - 1):
                if plot_colours2 == True:
                    plt.plot(time, out[i][2][j], label = r"$\alpha$" + str(j + 2).translate(SUB), color = colors[j])
                else:
                    plt.plot(time, out[i][2][j], label = r"$\alpha$" + str(j + 2).translate(SUB))
            plt.legend()
            plt.title(r"$\alpha$'s over Time for " + str(diff_swimmers[i][4]))
            plt.xlabel("Time (s)")
            plt.ylabel(r"$\alpha$ (rad)")
            plt.show()
    
    #Plotting First 2 Alphas
    
    if alphas_2D == True:
        for i in range(number_of_swimmers):
            plt.plot(out[i][2][0], out[i][2][1])
            plt.title("Phase Portrait for the " + str(diff_swimmers[i][4]))
            plt.xlabel(r"$\alpha$₂", fontsize = 15)
            plt.ylabel(r"$\alpha$₃", fontsize = 15)
            plt.show()
    
    pygame.quit()
    
    return out, positions

if __name__ == "__main__":
    time_start = time.time()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    out, positions = main(config)
    print("Time Taken:", time.time() - time_start)
