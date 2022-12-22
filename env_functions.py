import pygame
import numpy as np
from numpy import sin, cos

from run_config import eta, xi, black, rate, height, speed


def calculate_coords(x1, y1, theta, swimmer):
    swimmer.coords[0, 0] = x1
    swimmer.coords[1, 0] = y1
    swimmer.coords[0, 1:] = x1 + np.cumsum(swimmer.L * cos(theta))
    swimmer.coords[1, 1:] = y1 + np.cumsum(swimmer.L * sin(theta))


def construct_matrices(theta, coords, N, L, Q, P):
    for i in range(0, N - 1):
        Q[i + 1, 2:(3 + i)] = - L[:(i + 1)] * sin(theta[:(i + 1)])
        Q[i + 1 + N, 2:(3 + i)] = L[:(i + 1)] * cos(theta[:(i + 1)])

    change_x = coords[0, :N] - coords[0, 0]
    change_y = coords[1, :N] - coords[1, 0]

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


def RK4Addition(x, k1, k2, k3, k4, h):
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def draw_swimmer(swimmer, screen):
    calculate_coords(swimmer.x1, swimmer.y1, swimmer.theta, swimmer)
    for i in range(swimmer.N):
        pygame.draw.line(screen, black, (swimmer.coords[0, i], swimmer.coords[1, i]), (swimmer.coords[0, i + 1], swimmer.coords[1, i + 1]))
        if i > 0:
            pygame.draw.circle(screen, swimmer.dot_colour, (int(swimmer.coords[0, i]), int(swimmer.coords[1, i])), 2)


def gradient(x1, y1, theta0, swimmer):
    theta = np.append([theta0], swimmer.theta[1:])
    calculate_coords(x1, y1, theta, swimmer)
    M = construct_matrices(theta, swimmer.coords, swimmer.N, swimmer.L, swimmer.Q, swimmer.P)
    Nmat = M @ swimmer.Cinv
    A = Nmat[:, :3]
    B = Nmat[:, 3:]
    Ainv = np.linalg.inv(A)
    return - Ainv @ B @ swimmer.alphadot


def RK4(swimmer):
    h = 1 / rate
    xk1, yk1, tk1 = gradient(swimmer.x1, swimmer.y1, swimmer.theta[0], swimmer)
    xk2, yk2, tk2 = gradient(swimmer.x1 + h * xk1 / 2, swimmer.y1 + h * yk1 / 2, swimmer.theta[0] + h * tk1 / 2, swimmer)
    xk3, yk3, tk3 = gradient(swimmer.x1 + h * xk2 / 2, swimmer.y1 + h * yk2 / 2, swimmer.theta[0] + h * tk2 / 2, swimmer)
    xk4, yk4, tk4 = gradient(swimmer.x1 + h * xk3, swimmer.y1 + h * yk3, swimmer.theta[0] + h * tk3, swimmer)
    swimmer.x1 = RK4Addition(swimmer.x1, xk1, xk2, xk3, xk4, h)
    swimmer.y1 = RK4Addition(swimmer.y1, yk1, yk2, yk3, yk4, h)
    swimmer.theta[0] = RK4Addition(swimmer.theta[0], tk1, tk2, tk3, tk4, h)


# fixes the swimmer if it tries to overextend and sets the lambda values
def set_rates(swimmer, out):
    swimmer.alphadot = np.array(2 * out - 1)
    alpha = swimmer.alpha + swimmer.alphadot / rate
    swimmer.last_side = [-side if abs(alpha_i) >= swimmer.max_angle else side for side, alpha_i in zip(swimmer.last_side, alpha)]
    swimmer.alphadot = np.array([0 if abs(alpha_i) >= swimmer.max_angle else alphadot_i for alpha_i, alphadot_i in zip(alpha, swimmer.alphadot)])


def calculate_mid_dists(swimmer):
    if swimmer.N % 2:
        mid_joint = (swimmer.N // 2) + 1
        return swimmer.coords[0, mid_joint], swimmer.coords[1, mid_joint]

    else:
        lower_joint = swimmer.N // 2
        upper_joint = lower_joint + 1
        return (swimmer.coords[0, lower_joint] + swimmer.coords[0, upper_joint])/2, (swimmer.coords[1, lower_joint] + swimmer.coords[1, upper_joint])/2


# Calculates the x coordinate of the end of the swimmer.
def calculate_dist(swimmer):
    return swimmer.x1 + np.sum(swimmer.L * cos(swimmer.theta))


def calculate_y_dist(swimmer):
    return swimmer.y1 + np.sum(swimmer.L * sin(swimmer.theta))


# reports the fitness
def calculate_fitness(swimmer):
    return calculate_dist(swimmer) - swimmer.x_start


# sets up initial conditions
def set_up(swimmer, N, length, delta, number, total_number):
    swimmer.N = N
    swimmer.L = np.array([length for _ in range(N)])
    swimmer.delta = delta
    swimmer.max_angle = (delta / speed) / 2
    swimmer.x1 = 10
    swimmer.y1 = height * (number + 1) / (total_number + 1)
    swimmer.alphadot = np.zeros(N - 1)
    swimmer.x_start = calculate_dist(swimmer)
    swimmer.y_start = calculate_y_dist(swimmer)
    C = np.identity(N + 2)
    C[3:N + 2, 2:N + 1] += -np.identity(N - 1)
    swimmer.Cinv = np.linalg.inv(C).astype(int)

    swimmer.Q = np.zeros((N * 3, N + 2), dtype=float)
    swimmer.Q[:N, 0] = np.ones(N)
    swimmer.Q[N:2 * N, 1] = np.ones(N)
    swimmer.Q[2 * N:, 2:] = np.identity(N)

    swimmer.P = np.zeros((3, 3 * N), dtype=float)
    swimmer.coords = np.zeros((2, N + 1), dtype=float)
    calculate_coords(swimmer.x1, swimmer.y1, swimmer.theta, swimmer)

    swimmer.x_mid_start, swimmer.y_mid_start = calculate_mid_dists(swimmer)

    swimmer.dist = [0]
    swimmer.y_dist = [0]
    swimmer.alphas = np.array([swimmer.alpha])
    swimmer.positions = [swimmer.coords.copy()]


def report(swimmer):
    x_dist, y_dist = calculate_mid_dists(swimmer)
    swimmer.dist.append(x_dist - swimmer.x_mid_start)
    swimmer.y_dist.append(y_dist - swimmer.y_mid_start)
    swimmer.alphas = np.vstack((swimmer.alphas, swimmer.alpha))


def report_position(swimmer):
    calculate_coords(swimmer.x1, swimmer.y1, swimmer.theta, swimmer)
    swimmer.positions.append(swimmer.coords.copy())


def calculate_position(swimmer):
    RK4(swimmer)
    swimmer.alpha = swimmer.alpha + swimmer.alphadot / rate
    calculate_thetas(swimmer)


def calculate_thetas(swimmer):
    swimmer.theta[1:] = np.cumsum(swimmer.alpha) + swimmer.theta[0]


def iterate(swimmer, screen, tick_count):
    draw_swimmer(swimmer, screen)
    swimmer.calculate_alphadot(tick_count)
    calculate_position(swimmer)
    report(swimmer)
