import os
import neat

rate = 100
speed = 1

xi = 0.38*10**(-3)
eta = xi*1.89

width = 600
height = 600
black = (0, 0, 0)
white = (255, 255, 255)
red = (200, 0, 0)
blue = (0, 0, 200)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'neat_config')
neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

training_path = 'Training Output'
checkpoint_dir = 'Checkpoints'