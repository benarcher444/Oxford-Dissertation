# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 17:01:13 2021

@author: Ben Archer

Once a learner has been trained, move that winner.dat file to a folder. Then place that folder name
in the folders array and choose the rates at which you would like to simulate the swimmer at.

This file will calculate the positions of the swimmer at all following points for the desired runtime
and save the data in a file called data+rate.dat. This is then used by Plotting_Learner.py and
Visualising_Learner.py. This program has been optimised for speed and does not display the swimmers.
"""

import os

os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"
import neat
import pickle
import numpy as np
from numpy import sin, cos
import time

'''
To compile a swimmer, add its winner.dat file to a folder then add its name here.
'''

folders = [
    "4 - Link"
]

'''
The different rates you want to compile at. Rate is just 1/timestep. 
'''

rates = [30, 100, 1000]

'''
Different conditions for the swimmers. Make sure these are the same as the NEAT training file 
when you compile or the NEAT swimmer might not compile as desired.
'''

speed = 1
delta = 1.5 / speed
max_angle = delta / 2
runtime = 30

xi = 0.38 * 10 ** (-3)
eta = 1.89 * xi


def run(folder, rate, config_file):
    print("")
    print("Folder:", folder, "at rate:", rate)
    print("")

    # Sets up the genome of the winner.

    genome = pickle.load(open(folder + "//winner.dat", 'rb'))
    pickle.dump(genome, open(folder + "//winner.dat", 'wb'))

    N = int(calculate_N(genome) + 1)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    config.genome_config.num_inputs = 3 * N - 2
    config.genome_config.num_outputs = N - 1
    config.genome_config.input_keys = [-i for i in range(1, config.genome_config.num_inputs + 1)]
    config.genome_config.output_keys = [i for i in range(config.genome_config.num_outputs)]

    iterations = rate * runtime

    x_list = np.zeros(iterations + 1)
    y_list = np.zeros(iterations + 1)
    theta_list = np.zeros((iterations + 1, N))
    alpha_list = np.zeros((iterations + 1, N - 1))
    alpha_list = np.zeros((iterations + 1, N - 1))
    coords_list = np.zeros((iterations + 1, 2 * (N + 1)))

    # creates the swimmer

    swimmer = Swimmer(N, genome, config)

    coords_list[0, :] = swimmer.set_up()

    # iterates the swimmer and records its positions.

    for i in range(1, iterations + 1):
        x1, y1, theta, alpha, coords = swimmer.iterate()
        x_list[i] = x1
        y_list[i] = y1
        theta_list[i, :] = theta
        alpha_list[i, :] = alpha
        coords_list[i, :] = coords

        if i % (iterations / 10) == 0:
            print("Percentage Done:", int(i / (iterations) * 100))

    data = [x_list, y_list, theta_list, alpha_list, coords_list]

    return data


# Calculates N for the swimmer.

def calculate_N(genome):
    out = 0
    for i in range(200):
        try:
            genome.nodes[i]
            out += 1
        except:
            break
    return out


# The same class as in NEAT_Training

class Swimmer:
    def __init__(self, N, genome, config):
        self.x1 = 0
        self.y1 = 0
        self.N = N
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.last_side = np.zeros(N - 1) - 1
        self.theta = np.zeros(N)
        self.alpha = np.zeros(N - 1)
        self.alphadot = np.zeros(N - 1)
        self.L = np.array([20 for i in range(self.N)])

        C = np.identity(N + 2)
        C[3:N + 2, 2:N + 1] += -np.identity(N - 1)
        self.Cinv = np.linalg.inv(C).astype(int)

        self.Q = np.zeros((N * 3, N + 2), dtype=float)
        self.Q[:N, 0] = np.ones(N)
        self.Q[N:2 * N, 1] = np.ones(N)
        self.Q[2 * N:, 2:] = np.identity(N)

        self.P = np.zeros((3, 3 * N), dtype=float)
        self.coords = np.zeros((2, N + 1), dtype=float)
        self.flat_coords = np.zeros(2 * (N + 1), dtype=float)

    def set_up(self):
        self.calculate_flat_coords()
        return self.flat_coords

    def iterate(self):
        inputs = np.concatenate([[self.theta[0]], self.alpha, self.alphadot, self.last_side])
        out = np.array(self.net.activate(inputs))

        self.set_rates(out)
        self.RK4()

        self.alpha = self.alpha + self.alphadot / rate
        self.calculate_thetas()

        self.calculate_flat_coords()

        return self.x1, self.y1, self.theta, self.alpha, self.flat_coords

    def set_rates(self, out):
        self.alphadot = np.array(2 * out - 1)
        alpha = self.alpha + self.alphadot / rate
        self.last_side = [-self.last_side[i] if abs(alpha[i]) >= max_angle else self.last_side[i] for i in range(self.N - 1)]
        self.alphadot = np.array([0 if abs(alpha[i]) >= max_angle else self.alphadot[i] for i in range(self.N - 1)])

    def RK4(self):
        h = 1 / (rate)
        xk1, yk1, tk1 = self.gradient(self.x1, self.y1, self.theta[0])
        xk2, yk2, tk2 = self.gradient(self.x1 + h * xk1 / 2, self.y1 + h * yk1 / 2, self.theta[0] + h * tk1 / 2)
        xk3, yk3, tk3 = self.gradient(self.x1 + h * xk2 / 2, self.y1 + h * yk2 / 2, self.theta[0] + h * tk2 / 2)
        xk4, yk4, tk4 = self.gradient(self.x1 + h * xk3, self.y1 + h * yk3, self.theta[0] + h * tk3)
        self.x1 = self.RK4Addition(self.x1, xk1, xk2, xk3, xk4, h)
        self.y1 = self.RK4Addition(self.y1, yk1, yk2, yk3, yk4, h)
        self.theta[0] = self.RK4Addition(self.theta[0], tk1, tk2, tk3, tk4, h)

    def RK4Addition(self, x, k1, k2, k3, k4, h):
        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def gradient(self, x1, y1, theta0):
        theta = np.append([theta0], self.theta[1:])
        coords = self.calculate_coords(x1, y1, theta)
        M = self.construct_matrices(theta, coords)
        Nmat = M @ self.Cinv
        A = Nmat[:, :3]
        B = Nmat[:, 3:]
        Ainv = np.linalg.inv(A)
        return - Ainv @ B @ self.alphadot

    def construct_matrices(self, theta, coords):
        for i in range(0, self.N - 1):
            self.Q[i + 1, 2:(3 + i)] = - self.L[:(i + 1)] * sin(theta[:(i + 1)])
            self.Q[i + 1 + self.N, 2:(3 + i)] = self.L[:(i + 1)] * cos(theta[:(i + 1)])

        '''
        #An alternative method without the for loop but remarkably slower
        
        #Q[:N, 2:] = np.tril(- L*sin(theta), -1)
        #Q[N:(2*N), 2:] = np.tril(L*cos(theta), -1)
        
        #Q = np.block([[Q1, Q2, A],[Q2, Q1, B],[Q2, Q2, Q3]])
        '''

        change_x = coords[0, :self.N] - coords[0, 0]
        change_y = coords[1, :self.N] - coords[1, 0]

        self.P[0, :self.N] = self.L * (- xi * cos(theta) * cos(theta) - eta * sin(theta) * sin(theta))
        self.P[0, self.N:2 * self.N] = self.L * sin(theta) * cos(theta) * (-xi + eta)
        self.P[0, 2 * self.N:] = eta * self.L ** 2 / 2 * sin(theta)

        self.P[1, :self.N] = self.L * sin(theta) * cos(theta) * (-xi + eta)
        self.P[1, self.N:2 * self.N] = self.L * (- xi * sin(theta) ** 2 - eta * cos(theta) ** 2)
        self.P[1, 2 * self.N:] = - eta * self.L ** 2 / 2 * cos(theta)

        self.P[2, :self.N] = - change_x * self.L * sin(theta) * cos(theta) * (xi - eta) - change_y * self.L * (
                - xi * cos(theta) ** 2 - eta * sin(theta) ** 2) + eta * self.L ** 2 / 2 * sin(theta)
        self.P[2, self.N:2 * self.N] = - change_x * self.L * (xi * sin(theta) ** 2 + eta * cos(theta) ** 2) - change_y * self.L * sin(theta) * cos(theta) * (
                - xi + eta) - eta * self.L ** 2 / 2 * cos(theta)
        self.P[2, 2 * self.N:] = - change_x * eta * self.L ** 2 / 2 * cos(theta) - change_y * eta * self.L ** 2 / 2 * sin(theta) - eta * self.L ** 3 / 3

        return self.P @ self.Q

    def calculate_coords(self, x, y, theta):
        self.coords[0, 0] = x
        self.coords[1, 0] = y
        self.coords[0, 1:] = x + np.cumsum(self.L * cos(theta))
        self.coords[1, 1:] = y + np.cumsum(self.L * sin(theta))
        return self.coords

    def calculate_flat_coords(self):
        self.flat_coords[0] = self.x1
        self.flat_coords[1:(self.N + 1)] = self.x1 + np.cumsum(self.L * cos(self.theta))
        self.flat_coords[self.N + 1] = self.y1
        self.flat_coords[(self.N + 2):] = self.y1 + np.cumsum(self.L * sin(self.theta))

    def calculate_thetas(self):
        self.alpha[0] += self.theta[0]
        self.theta[1:] = np.cumsum(self.alpha)
        self.alpha[0] += - self.theta[0]

    def calculate_dist(self):
        return self.x1 + np.sum(self.L * cos(self.theta))


if __name__ == '__main__':
    start_time = time.time()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    for rate in rates:
        for folder in folders:
            data = run(folder, rate, config_path)
            pickle.dump(data, open(folder + '//data' + str(rate) + '.dat', 'wb'))
    print("Time Taken:", time.time() - start_time)
