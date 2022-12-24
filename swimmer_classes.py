import neat
import numpy as np

from env_functions import set_rates, calculate_thetas, set_up
from run_config import red, blue, speed


class Learning_Swimmer:
    # Initialises each swimmer.
    def __init__(self, N, length, delta, rate, number, total_number, genome, neat_config):
        self.dot_colour = red
        self.theta = np.zeros(N)
        self.alpha = np.zeros(N - 1)

        self.last_side = np.zeros(N - 1) - 1
        self.net = neat.nn.FeedForwardNetwork.create(genome, neat_config)

        set_up(self, N, length, delta, rate, number, total_number)

    # Moves the swimmer one timestep
    def calculate_alphadot(self, _):
        inputs = np.concatenate([[self.theta[0]], self.alpha, self.alphadot, self.last_side])
        out = np.array(self.net.activate(inputs))
        set_rates(self, out)


class Purcell_Swimmer:
    def __init__(self, N, length, delta, rate, number, total_number):
        self.alphadot = None
        self.dot_colour = blue
        self.theta = np.zeros(N)
        if N % 2 == 0:
            self.theta[0] = delta * speed / 2 * (N // 2 - 1) + delta * speed / 4
        else:
            self.theta[0] = delta * speed / 2 * (N // 2)

        self.alpha = np.zeros(N - 1) - delta * speed / 2
        calculate_thetas(self)

        set_up(self, N, length, delta, rate, number, total_number)

    def calculate_alphadot(self, tick_count):
        time_elapsed = tick_count / self.rate
        which = int((time_elapsed // self.delta) % (2 * (self.N - 1)))
        negative = 1 if which < (self.N - 1) else -1
        which = which % (self.N - 1)
        alphadot = np.zeros(self.N - 1)
        alphadot[which] = negative * speed

        self.alphadot = np.flip(alphadot)


class Test1_Swimmer:
    def __init__(self, N, length, delta, rate, number, total_number):
        self.alphadot = None
        self.dot_colour = blue
        self.theta = np.append(np.zeros(N - 1), [-delta * speed / 2])
        self.alpha = self.theta[1:N] - self.theta[:(N - 1)]

        set_up(self, N, length, delta, rate, number, total_number)

    def calculate_alphadot(self, tick_count):
        time_elapsed = tick_count / self.rate
        if time_elapsed % (self.delta * 2) < self.delta:
            self.alphadot = np.append(np.zeros(self.N - 2), [1])
        else:
            self.alphadot = np.append(np.zeros(self.N - 2), [-1])


class Test2_Swimmer:
    def __init__(self, N, length, delta, rate, number, total_number):
        self.alphadot = None
        self.dot_colour = blue
        self.theta = np.zeros(N)
        self.alpha = self.theta[1:N] - self.theta[:(N - 1)]

        set_up(self, N, length, delta, rate, number, total_number)

    def calculate_alphadot(self, tick_count):
        time_elapsed = tick_count / self.rate
        if time_elapsed % (self.delta * 1.5) < self.delta:
            self.alphadot = np.zeros(self.N - 1) + 0.5
        else:
            self.alphadot = np.zeros(self.N - 1) - 1

