# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 16:03:37 2021

@author: Ben Archer

This file plots previously compiled learners. This is the quickest way to make plots as the
simulation does not have to be run again. Place the folder name in the folders array then 
select which plots you would like to produce
"""

import pickle
import matplotlib.pyplot as plt
import time
import numpy as np

folders = [
         "3 - Link 𝛿 = 1.5"
        ,"4 - Link"
        ,"5 - Link"
        ,"6 - Link"
        ,"10 - Link"
         ]

'''
Rate: Which timestep results you would like to plot
Number of shots: The number of total images of the swimmer you would like to overlap
Occ Number of shots: The number of total images of the swimmer you would like to overlap
                     between lower and upper
                     
Lower: The minimum time for the alphas plot
Upper: The maximum time for the alphas plot
Spacing: The spacing betwen each spaced placement plot
'''

rate = 1000
number_of_shots = 100
occ_number_of_shots = 50
lower = 0
upper = 10
positions = np.linspace(lower/30, upper/30, 8)
spacing = 200

x_displacement = True
placements = True
occ_placements = True
spaced_placements =  True
alphas = True
alphas_compared_2D = True
alphas_compared_3D = True

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

class Swimmer():
    def __init__(self, folder):
        data = pickle.load(open(folder + '//data' + str(rate) + '.dat', 'rb'))
        pickle.dump(data, open(folder + '//data' + str(rate) + '.dat', 'wb'))
        self.name = folder
        self.alphas = data[3]
        self.coords_list = data[4]
        self.iterations = len(self.alphas)
        self.N = int(len(self.coords_list[0])/2 - 1)
        self.xs = np.linspace(0, self.iterations - 1, self.iterations)/rate
    
    def distance(self):
        middle = int(self.N//2)
        if self.N % 2 == 0:
            distances = self.coords_list[:, middle]
        else:
            distances = (self.coords_list[:, middle] + self.coords_list[:, middle + 1])/2
        
        distances = distances - distances[0]
        return distances, self.name, self.xs
    
    def placements(self):
        col_change = [0, 0, 1]
        coord_keys = np.round(np.linspace(0, self.iterations - 1, num = number_of_shots)).astype(int)
        
        for key in coord_keys:
            coords = self.coords_list[key]
            xs = coords[:(self.N+1)]
            ys = coords[(self.N+1):]
            plt.plot(xs, ys, color = tuple(col_change))
            col_change[0] += 1/(number_of_shots + 2)
            col_change[2] += -1/(number_of_shots + 2)
            
        plt.title("Placement of " + self.name + " over Time")
        plt.xlabel(r'$x$ (μm)')
        plt.ylabel(r"$y$ (μm)")
        plt.gca().set_aspect("equal")
        plt.show()
        
    def occ_placements(self):
        coord_keys = np.round(np.linspace(lower/30*(self.iterations-1), upper/30*(self.iterations-1), num = occ_number_of_shots)).astype(int)
        
        for i, key in enumerate(coord_keys):
            col_change = [0, 0, 1]
            col_change[0] += (1/(self.iterations))*(key - lower/30*(self.iterations-1))/(upper - lower)*30
            col_change[2] += (-1/(self.iterations))*(key - lower/30*(self.iterations-1))/(upper - lower)*30
            coords = self.coords_list[key]
            xs = coords[:(self.N+1)]
            ys = coords[(self.N+1):]
            plt.plot(xs, ys, color = tuple(col_change))
            
        plt.title("Placement of " + self.name + " over Time")
        plt.xlabel(r'$x$ (μm)')
        plt.ylabel(r"$y$ (μm)")
        plt.gca().set_aspect("equal")
        plt.show()
        
    def spaced_placements(self):
        shots = ((self.iterations-1)*positions)
        for i, shot in enumerate(shots):
            col_change = [0, 0, 1]
            col_change[0] += (1/(self.iterations))*(shot - lower/30*(self.iterations-1))/(upper - lower)*30
            col_change[2] += (-1/(self.iterations))*(shot - lower/30*(self.iterations-1))/(upper - lower)*30
            coords = self.coords_list[int(shot)]
            xs = coords[:(self.N+1)]
            ys = coords[(self.N+1):]
            xs += i*spacing
            plt.ylim([-80, 80])
            plt.plot(xs, ys, color = tuple(col_change))
        plt.gca().set_aspect("equal")
        plt.show()
      
    def alpha_plot(self):
        alphas = np.rollaxis(self.alphas, 1)
        for i, alpha in enumerate(alphas):
            plt.plot(self.xs[int(lower*rate):int(upper*rate)], alpha[int(lower*rate):int(upper*rate)], label = r"$\alpha$" + str(i + 2).translate(SUB))
        plt.legend()
        plt.title("Alpha Values over Time for the " + self.name)
        plt.xlabel("Time (s)")
        plt.ylabel(r"$\alpha$ (rad)")
        plt.show()
    
    def alpha_compare_2D(self):
        alphas = np.rollaxis(self.alphas, 1)
        number_of_alphas = len(alphas)
        for i in range(number_of_alphas - 1):
            plt.plot(-alphas[i], alphas[i + 1])
            plt.xlabel(r"-$\alpha$" + str(i + 2).translate(SUB) + " (rad)", fontsize = 15)
            plt.ylabel(r"$\alpha$" + str(i + 3).translate(SUB) + " (rad)", fontsize = 15)
            plt.gca().set_aspect("equal")
            plt.show()
            
    def alpha_compare_3D(self):
        alphas = np.rollaxis(self.alphas, 1)
        number_of_alphas = len(alphas)
        if number_of_alphas > 2:
            for i in range(number_of_alphas - 2):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(xs = alphas[i], ys = alphas[i + 1], zs = alphas[i + 2])
                ax.set_xlabel(r"$\alpha$" + str(i + 2).translate(SUB) + " (rad)")
                ax.set_ylabel(r"$\alpha$" + str(i + 3).translate(SUB) + " (rad)")
                ax.set_zlabel(r"$\alpha$" + str(i + 4).translate(SUB) + " (rad)")
                plt.show()
        
def run():
    number_of_swimmers = len(folders)
    swimmers = []
    
    for i in range(number_of_swimmers):
        swimmers.append(Swimmer(folders[i]))
    
    if x_displacement == True:
        for swimmer in swimmers:
            distance, name, xs = swimmer.distance()
            print(name, "Distance :", distance[-1])
            plt.plot(xs, distance, label = name)
        plt.legend()
        plt.title("Displacement of Swimmers over Time")
        plt.xlabel("Time (s)")
        plt.ylabel(r"$x$ Displacement (μm)")
        plt.show()
    
    if placements == True:
        for swimmer in swimmers:
            swimmer.placements()
            
    if occ_placements == True:
        for swimmer in swimmers:
            swimmer.occ_placements()
            
    if spaced_placements == True:
        for swimmer in swimmers:
            swimmer.spaced_placements()
            
    if alphas == True:
        for swimmer in swimmers:
            swimmer.alpha_plot()
    
    if alphas_compared_2D == True:
        for swimmer in swimmers:
            swimmer.alpha_compare_2D()
            
    if alphas_compared_3D == True:
        for swimmer in swimmers:
            swimmer.alpha_compare_3D()
    
if __name__ == '__main__':
    start_time = time.time()
    run()
    print("Time Taken:", time.time() - start_time)
