import neat
import numpy as np
from numpy import sin, cos
from typing import Tuple

# Import functions and variables from other modules
from helper_functions import RK4Addition
from run_config import red, blue, speed, height, xi, eta


class Swimmer:
    alpha = None
    theta = None
    dot_colour = None

    def __init__(self, N, length, delta, rate, number, total_number):
        """
            Set up the initial conditions for a swimmer object.

            Parameters:
                self (object): The swimmer object to set up the initial conditions for.
                N (int): The number of joints in the swimmer object.
                length (float): The length of each joint in the swimmer object.
                delta (float): The maximum angle of movement for each joint in the swimmer object.
                rate (int): The rate at which the swimmer object is updated.
                number (int): The index of the swimmer object.
                total_number (int): The total number of swimmer objects.
            """

        # Set initial conditions for swimmer object
        self.N = N
        self.rate = rate
        self.L = np.array([length for _ in range(N)])
        self.delta = delta
        self.max_angle = (delta / speed) / 2
        self.x1 = 10
        self.y1 = height * (number + 1) / (total_number + 1)
        self.alphadot = np.zeros(N - 1)
        self.x_start = self.calculate_x_dist()
        self.y_start = self.calculate_y_dist()

        # Set up matrices for swimmer object
        C = np.identity(N + 2)
        C[3:N + 2, 2:N + 1] += -np.identity(N - 1)
        self.Cinv = np.linalg.inv(C).astype(int)

        self.Q = np.zeros((N * 3, N + 2), dtype=float)
        self.Q[:N, 0] = np.ones(N)
        self.Q[N:2 * N, 1] = np.ones(N)
        self.Q[2 * N:, 2:] = np.identity(N)

        self.P = np.zeros((3, 3 * N), dtype=float)

        # Calculate initial coordinates for swimmer object
        self.coords = np.zeros((2, N + 1), dtype=float)
        self.calculate_coords(self.x1, self.y1, self.theta)

        # Calculate initial midpoint coordinates for swimmer object
        self.x_mid_start, self.y_mid_start = self.calculate_mid_dists()

        # Initialize lists to store distance and angle information for swimmer object
        self.x_dist = [0]
        self.y_dist = [0]
        self.alphas = np.array([self.alpha])
        self.thetas = np.array([self.theta])
        self.coords_list = [self.coords.copy()]

    def calculate_position(self):
        """
        Calculates the new position of the swimmer using the Runge-Kutta method.
        Updates alphas and recalculates thetas.
        """
        self.RK4()
        self.alpha = self.alpha + self.alphadot / self.rate
        self.calculate_thetas()

    def RK4(self):
        """
        Perform a single step of a fourth-order Runge-Kutta integration method on the swimmer object.
        """

        # Calculate time step
        h = 1 / self.rate

        # Calculate gradient at current time step
        xk1, yk1, tk1 = self.gradient(self.x1, self.y1, self.theta[0])

        # Calculate gradient at midpoint of interval
        xk2, yk2, tk2 = self.gradient(self.x1 + h * xk1 / 2, self.y1 + h * yk1 / 2, self.theta[0] + h * tk1 / 2)

        # Calculate gradient at midpoint of interval
        xk3, yk3, tk3 = self.gradient(self.x1 + h * xk2 / 2, self.y1 + h * yk2 / 2, self.theta[0] + h * tk2 / 2)

        # Calculate gradient at end of interval
        xk4, yk4, tk4 = self.gradient(self.x1 + h * xk3, self.y1 + h * yk3, self.theta[0] + h * tk3)

        # Update x1, y1, and theta[0] using Runge-Kutta 4th order integration method
        self.x1 = RK4Addition(self.x1, xk1, xk2, xk3, xk4, h)
        self.y1 = RK4Addition(self.y1, yk1, yk2, yk3, yk4, h)
        self.theta[0] = RK4Addition(self.theta[0], tk1, tk2, tk3, tk4, h)

    def gradient(self, x1: float, y1: float, theta0: float) -> np.ndarray:
        """
        Calculate the gradient of x1, y1 and theta0 for a given swimmer given a certain x1, y1, and theta0

        Parameters:
            x1 (float): The x-coordinate of the first point on the swimmer object.
            y1 (float): The y-coordinate of the first point on the swimmer object.
            theta0 (float): The angle between the x-axis and the first segment of the swimmer object.

        Returns:
            np.ndarray: The gradient of the swimmer object.
        """

        # Append theta0 to the swimmer's theta values
        theta = np.append([theta0], self.theta[1:])

        # Calculate coordinates of swimmer object
        self.calculate_coords(x1, y1, theta)

        # Calculate matrices Q and P
        M = self.construct_matrices(theta, self.N, self.L, self.Q, self.P)

        # Calculate N matrix
        Nmat = M @ self.Cinv

        # Split N matrix into A and B matrices
        A = Nmat[:, :3]
        B = Nmat[:, 3:]

        # Calculate inverse of A matrix
        Ainv = np.linalg.inv(A)

        # Calculate gradient
        return - Ainv @ B @ self.alphadot

    def construct_matrices(self, theta: np.ndarray, N: int, L: np.ndarray, Q: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Construct matrices Q and P and return their product.

        Parameters:
            theta (np.ndarray): An array of angles between the x-axis and the swimmer object.
            N (int): The number of points in the swimmer object.
            L (np.ndarray): An array of the lengths of the segments in the swimmer object.
            Q (np.ndarray): An array to store matrix Q.
            P (np.ndarray): An array to store matrix P.

        Returns:
            np.ndarray: The product of matrices P and Q.
        """

        # Construct matrices Q and P
        for i in range(0, N - 1):
            Q[i + 1, 2:(3 + i)] = - L[:(i + 1)] * sin(theta[:(i + 1)])
            Q[i + 1 + N, 2:(3 + i)] = L[:(i + 1)] * cos(theta[:(i + 1)])

        change_x = self.coords[0, :N] - self.coords[0, 0]
        change_y = self.coords[1, :N] - self.coords[1, 0]

        P[0, :N] = L * (- xi * cos(theta) * cos(theta) - eta * sin(theta) * sin(theta))
        P[0, N:2 * N] = L * sin(theta) * cos(theta) * (-xi + eta)
        P[0, 2 * N:] = eta * L ** 2 / 2 * sin(theta)

        P[1, :N] = L * sin(theta) * cos(theta) * (-xi + eta)
        P[1, N:2 * N] = L * (- xi * sin(theta) ** 2 - eta * cos(theta) ** 2)
        P[1, 2 * N:] = - eta * L ** 2 / 2 * cos(theta)

        P[2, :N] = - change_x * L * sin(theta) * cos(theta) * (xi - eta) - change_y * L * (- xi * cos(theta) ** 2 - eta * sin(theta) ** 2) + eta * L ** 2 / 2 * sin(theta)
        P[2, N:2 * N] = - change_x * L * (xi * sin(theta) ** 2 + eta * cos(theta) ** 2) - change_y * L * sin(theta) * cos(theta) * (- xi + eta) - eta * L ** 2 / 2 * cos(theta)
        P[2, 2 * N:] = - change_x * eta * L ** 2 / 2 * cos(theta) - change_y * eta * L ** 2 / 2 * sin(theta) - eta * L ** 3 / 3

        return P @ Q

    def calculate_fitness(self) -> float:
        """
        Calculate the fitness of the swimmer object.

        Returns:
            float: The fitness of the swimmer object.
        """

        # Calculate fitness of swimmer object by comparing the current distance to the start position
        return self.calculate_x_dist() - self.x_start

    def calculate_coords(self, x1: float, y1: float, theta: np.array):
        """
        Calculate the coordinates for a swimmer object.

        Parameters:
            x1 (float): The x-coordinate of the first point of the swimmer object.
            y1 (float): The y-coordinate of the first point of the swimmer object.
            theta (float or numpy array): The angle(s) between the x-axis and the swimmer object.
        """

        # Set first two coordinates of swimmer object
        self.coords[0, 0] = x1
        self.coords[1, 0] = y1

        # Set remaining coordinates of swimmer object using cumulative sum of L and theta
        self.coords[0, 1:] = x1 + np.cumsum(self.L * cos(theta))
        self.coords[1, 1:] = y1 + np.cumsum(self.L * sin(theta))

    def calculate_thetas(self):
        """
        Calculates updated theta values from the alpha values
        """
        self.theta[1:] = np.cumsum(self.alpha) + self.theta[0]

    def calculate_mid_dists(self) -> Tuple[float, float]:
        """
        Calculate the x and y coordinates of the midpoint of the swimmer object.
        If it is an odd number it takes the middle of the middle paddle. If even it takes the middle joint.

        Returns:
            tuple: A tuple containing the x and y coordinates of the midpoint.
        """

        # If the number of joints in the swimmer object is odd, the midpoint is the coordinate of the joint in the middle
        if self.N % 2:
            mid_joint = (self.N // 2) + 1
            return self.coords[0, mid_joint], self.coords[1, mid_joint]

        # If the number of joints in the swimmer object is even, the midpoint is the average of the coordinates of the two middle joints
        else:
            lower_joint = self.N // 2
            upper_joint = lower_joint + 1
            return (self.coords[0, lower_joint] + self.coords[0, upper_joint]) / 2, (self.coords[1, lower_joint] + self.coords[1, upper_joint]) / 2

    def calculate_x_dist(self) -> float:
        """
        Calculate the x coordinate of the end of the swimmer object.

        Returns:
            float: The x coordinate of the end of the swimmer object.
        """

        # Calculate x coordinate of end of swimmer object
        return self.x1 + np.sum(self.L * cos(self.theta))

    def calculate_y_dist(self) -> float:
        """
        Calculate the y coordinate of the end of the swimmer object.

        Returns:
            float: The y coordinate of the end of the swimmer object.
        """

        # Calculate y coordinate of end of swimmer object
        return self.y1 + np.sum(self.L * sin(self.theta))

    def report(self):
        """
        A way of recording the swimmers placements and progress for graphing/compiling for future use.
        """
        # Calculate the x and y distances of the midpoint of the swimmer object from its starting position
        x_dist, y_dist = self.calculate_mid_dists()

        # Append the x and y distances to the x_dist and y_dist lists, respectively
        self.x_dist.append(x_dist - self.x_mid_start)
        self.y_dist.append(y_dist - self.y_mid_start)

        # Update the alphas and thetas arrays with the current alpha and theta values of the swimmer object
        self.alphas = np.vstack((self.alphas, self.alpha))
        self.thetas = np.vstack((self.thetas, self.theta))

        # Calculate the current coordinates of the swimmer object
        self.calculate_coords(self.x1, self.y1, self.theta)

        # Add a copy of the swimmer object's current coordinates to the coords_list list
        self.coords_list.append(self.coords.copy())


# A class for swimmers that learn through the neat algorithm. The main difference being the use of a neural net for policy.
class Learning_Swimmer(Swimmer):
    # Initialises each swimmer.
    def __init__(self, N, length, delta, rate, number, total_number, genome, neat_config):
        # Set swimmer properties
        self.dot_colour = red
        self.theta = np.zeros(N)
        self.alpha = np.zeros(N - 1)

        self.last_side = np.zeros(N - 1) - 1
        self.net = neat.nn.FeedForwardNetwork.create(genome, neat_config)

        # Set up swimmer
        super().__init__(N, length, delta, rate, number, total_number)

    # Calculates rate of change of alpha for this time step.
    def calculate_alphadot(self, _):
        """
        Set the angular velocities and angles of the swimmer object. It makes sure that the angles do not overextend.
        """
        # Calculate inputs and output from neural network
        inputs = np.concatenate([[self.theta[0]], self.alpha, self.alphadot, self.last_side])
        out = np.array(self.net.activate(inputs))

        # Set angular velocity of swimmer object
        self.alphadot = np.array(2 * out - 1)

        # Calculate new angle values
        alpha = self.alpha + self.alphadot / self.rate

        # Reverse direction of swimmer object if angle exceeds max_angle
        self.last_side = [-side if abs(alpha_i) >= self.max_angle else side for side, alpha_i in zip(self.last_side, alpha)]

        # Set angular velocity to 0 if angle exceeds max_angle
        self.alphadot = np.array([0 if abs(alpha_i) >= self.max_angle else alphadot_i for alpha_i, alphadot_i in zip(alpha, self.alphadot)])


# Class for Purcell swimmers
class Purcell_Swimmer(Swimmer):
    def __init__(self, N, length, delta, rate, number, total_number):
        self.alphadot = None
        self.dot_colour = blue
        self.theta = np.zeros(N)

        # Set initial theta values
        if N % 2 == 0:
            self.theta[0] = delta * speed / 2 * (N // 2 - 1) + delta * speed / 4
        else:
            self.theta[0] = delta * speed / 2 * (N // 2)

        self.alpha = np.zeros(N - 1) - delta * speed / 2

        # Calculate theta values based on alpha values
        super().calculate_thetas()

        # Set up swimmer
        super().__init__(N, length, delta, rate, number, total_number)

    # Calculates rate of change of alpha for this time step.
    def calculate_alphadot(self, tick_count):
        # Calculate current time
        time_elapsed = tick_count / self.rate
        # Calculate current segment
        which = int((time_elapsed // self.delta) % (2 * (self.N - 1)))
        # Determine direction of rotation for current segment
        negative = 1 if which < (self.N - 1) else -1
        which = which % (self.N - 1)
        alphadot = np.zeros(self.N - 1)
        # Set angular rate for current segment
        alphadot[which] = negative * speed

        self.alphadot = np.flip(alphadot)


# Class for test swimmers that move in a wave pattern.
class Test1_Swimmer(Swimmer):
    def __init__(self, N, length, delta, rate, number, total_number):
        self.alphadot = None
        self.dot_colour = blue
        # Set initial theta values
        self.theta = np.append(np.zeros(N - 1), [-delta * speed / 2])
        # Calculate initial alpha values based on theta values
        self.alpha = self.theta[1:N] - self.theta[:(N - 1)]

        # Set up swimmer
        super().__init__(N, length, delta, rate, number, total_number)

    # Calculates rate of change of alpha for this time step.
    def calculate_alphadot(self, tick_count):
        # Calculate current time
        time_elapsed = tick_count / self.rate
        # Set angular rates based on current time
        if time_elapsed % (self.delta * 2) < self.delta:
            self.alphadot = np.append(np.zeros(self.N - 2), [1])
        else:
            self.alphadot = np.append(np.zeros(self.N - 2), [-1])


# Class for swimmers that move in a scallop pattern.
class Test2_Swimmer(Swimmer):
    def __init__(self, N, length, delta, rate, number, total_number):
        self.alphadot = None
        self.dot_colour = blue
        self.theta = np.zeros(N)
        # Calculate initial alpha values based on theta values
        self.alpha = self.theta[1:N] - self.theta[:(N - 1)]
        # Set up swimmer
        super().__init__(N, length, delta, rate, number, total_number)

    # Calculates rate of change of alpha for this time step.
    def calculate_alphadot(self, tick_count):
        # Calculate current time
        time_elapsed = tick_count / self.rate
        # Set angular rates based on current time
        if time_elapsed % (self.delta * 1.5) < self.delta:
            self.alphadot = np.zeros(self.N - 1) + 0.5
        else:
            self.alphadot = np.zeros(self.N - 1) - 1
