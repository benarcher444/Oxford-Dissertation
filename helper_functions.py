# Import required modules
import pygame
import os
from typing import List, Dict, Any
import pickle

# Import configuration variables from run_config module
from run_config import black, compiled_dir


def RK4Addition(x: float, k1: float, k2: float, k3: float, k4: float, h: float) -> float:
    """
    Perform a Runge-Kutta 4th order integration step.

    Parameters:
        x (float): The current value of the quantity being integrated.
        k1 (float): The value of the derivative at the beginning of the interval.
        k2 (float): The value of the derivative at the midpoint of the interval.
        k3 (float): The value of the derivative at the midpoint of the interval.
        k4 (float): The value of the derivative at the end of the interval.
        h (float): The size of the time interval.

    Returns:
        float: The updated value of the quantity being integrated.
    """

    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def draw_swimmer(swimmer, screen: pygame.SurfaceType):
    """
    Draw the swimmer object on the screen in real time.

    Parameters:
        swimmer (object): The swimmer object to be drawn.
        screen (pygame.SurfaceType): The surface to draw the swimmer object on.
    """

    # Calculate coordinates of swimmer object
    swimmer.calculate_coords(swimmer.x1, swimmer.y1, swimmer.theta)

    # Draw lines between points on swimmer object
    for i in range(swimmer.N):
        pygame.draw.line(screen, black, (swimmer.coords[0, i], swimmer.coords[1, i]), (swimmer.coords[0, i + 1], swimmer.coords[1, i + 1]))

        # Draw a dot at each point on swimmer object
        if i > 0:
            pygame.draw.circle(screen, swimmer.dot_colour, (int(swimmer.coords[0, i]), int(swimmer.coords[1, i])), 2)


def draw_compiled_swimmer(swimmer, screen: pygame.SurfaceType, tick: int):
    """
    Draw a compiled swimmer object on the screen at a specific time tick.

    Parameters:
        swimmer (object): The compiled swimmer object to be drawn.
        screen (pygame.SurfaceType): The surface to draw the swimmer object on.
        tick (int): The time tick to draw the swimmer object at.
    """

    # Draw lines between points on swimmer object
    for i in range(swimmer.N):
        pygame.draw.line(screen, black, (swimmer.coords_list[tick][0, i], swimmer.coords_list[tick][1, i]),
                         (swimmer.coords_list[tick][0, i + 1], swimmer.coords_list[tick][1, i + 1]))

        # Draw a dot at each point on swimmer object
        if i > 0:
            pygame.draw.circle(screen, swimmer.dot_colour, (int(swimmer.coords_list[tick][0, i]), int(swimmer.coords_list[tick][1, i])), 2)


def set_up_config(config: object, N: int) -> object:
    """
    Set up the genome configuration for a given config object and N.

    Parameters:
        config: config object to be modified
        N: integer value to be used in setting up the genome configuration

    Returns:
        The modified config object
    """

    # Set the number of inputs and outputs for the genome
    config.genome_config.num_inputs = 3 * N - 2
    config.genome_config.num_outputs = N - 1

    # Set the keys for the inputs and outputs of the genome
    config.genome_config.input_keys = [-i for i in range(1, config.genome_config.num_inputs + 1)]
    config.genome_config.output_keys = [i for i in range(config.genome_config.num_outputs)]

    return config


def calculate_N(genome: object) -> int:
    """
    Calculate the value of N for a given genome.

    Parameters:
    genome: object representing a genome

    Returns:
    Integer value of N
    """
    nodes = list(genome.nodes.keys())
    missing = next((a for (a, b) in zip(nodes, nodes[1:]) if a + 1 != b), None)
    return nodes[missing] + 2 if missing is not None else nodes[-1] + 2


def save_data(swimmer, swimmer_config: Dict[str, Any], rate: float, runtime: float):
    """
    Save data for a swimmer. Saves it to a file in the compiling directory in a folder taken from the compile field.

    Parameters:
        swimmer: object representing a swimmer
        swimmer_config: dictionary containing swimmer configuration data
        rate: float representing the rate at which data was collected
        runtime: float representing the total runtime for data collection
    """
    folder = os.path.join(compiled_dir, swimmer_config['compile'])
    if not os.path.exists(folder):
        os.mkdir(folder)

    path = os.path.join(folder, f'compiled-rate-{rate}-runtime-{runtime}.dat')
    print(f"Compiling to {path}")

    y_adjust(swimmer)

    data = {'x_dist': swimmer.x_dist, 'y_dist': swimmer.y_dist, 'thetas': swimmer.thetas, 'label': swimmer_config['label'],
            'alphas': swimmer.alphas, 'coords_list': swimmer.coords_list, 'dot_colour': swimmer.dot_colour}

    pickle.dump(data, open(path, 'wb'))


def y_adjust(swimmer):
    """
    Adjust the y-coordinates of a swimmer object before compiling.
    This is necessary as when the swimmer runs its y-coord is scaled so that it fits on the screen.
    Ideally, we want the swimmers to all have y0 = 0.

    Parameters:
        swimmer: object representing a swimmer whose y-coordinates will be adjusted
    """
    y_start = swimmer.coords_list[0][1, 0]

    for i, coords in enumerate(swimmer.coords_list):
        coords[1, :] = coords[1, :] - y_start


def compile_swimmers(swimmers: List, swimmer_creation: List[Dict[str, Any]], rate: float, runtime: float):
    """
    Compile data for a list of swimmers. Checks if there is a compile field on the array.
    If so runs save_data to save the swimmmer.

    Parameters:
        swimmers: list of objects representing swimmers
        swimmer_creation: list of dictionaries containing swimmer configuration data
        rate: float representing the rate at which data was collected
        runtime: float representing the total runtime for data collection
    """
    if not os.path.exists(compiled_dir):
        os.mkdir(compiled_dir)

    for swimmer, swimmer_config in zip(swimmers, swimmer_creation):
        if swimmer_config.get('compile', None):
            save_data(swimmer, swimmer_config, rate, runtime)


def iterate_and_report(swimmer, screen: pygame.surface, tick_count: int):
    """
    Draws the swimmer on the screen, calculates the new swimmer angle values,
    calculates the new position of the swimmer, and reports the progress of the swimmer.
    Parameters:
        swimmer (object): The swimmer object
        screen (pygame.Surface): The pygame surface to draw the swimmer on
        tick_count (int): The current tick count
    """
    draw_swimmer(swimmer, screen)
    swimmer.calculate_alphadot(tick_count)
    swimmer.calculate_position()
    swimmer.report()


def iterate(swimmer, screen: pygame.surface, tick_count: int):
    """
    Draws the swimmer on the screen, calculates the new swimmer angle values,
    calculates the new position of the swimmer, but doesn't report the swimmer positions.

    Parameters:
        swimmer (object): The swimmer object
        screen (pygame.Surface): The pygame surface to draw the swimmer on
        tick_count (int): The current tick count
    """
    draw_swimmer(swimmer, screen)
    swimmer.calculate_alphadot(tick_count)
    swimmer.calculate_position()
