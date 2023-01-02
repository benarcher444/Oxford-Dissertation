import os
import neat

# Set simulation parameters
speed = 1
xi = 0.38*10**(-3)
eta = xi*1.89

# Set display parameters
width = 600
height = 600
black = (0, 0, 0)
white = (255, 255, 255)
red = (200, 0, 0)
blue = (0, 0, 200)
space_between_positions = 80
number_of_overlays = 10

# Set directories for output
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'neat_config')

# Load NEAT configuration
neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

# Set directory names
training_dir = 'Training Output'
checkpoint_dir = 'Checkpoints'
compiled_dir = 'Compiled Runs'
