# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 15:50:44 2021

@author: Ben Archer
"""
from __future__ import print_function

import copy
import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np

from run_config import space_between_positions
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_spikes(spikes, view=False, filename=None, title=None):
    """ Plots the trains for a single spiking neuron. """
    t_values = [t for t, I, v, u, f in spikes]
    v_values = [v for t, I, v, u, f in spikes]
    u_values = [u for t, I, v, u, f in spikes]
    I_values = [I for t, I, v, u, f in spikes]
    f_values = [f for t, I, v, u, f in spikes]

    fig = plt.figure()
    plt.subplot(4, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    if title is None:
        plt.title("Izhikevich's spiking neuron model")
    else:
        plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(4, 1, 2)
    plt.ylabel("Fired")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, f_values, "r-")

    plt.subplot(4, 1, 3)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(4, 1, 4)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig


def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot


def plot_x_dist(swimmers, swimmer_creation, total_ticks):
    for swimmer_config, swimmer in zip(swimmer_creation, swimmers):
        time_space = np.linspace(0, total_ticks - 1, total_ticks) / swimmer.rate
        print(swimmer_config['label'])
        plt.plot(time_space, swimmer.x_dist, label=swimmer_config['label'])
        print(f"{swimmer_config['label']}: {swimmer.x_dist[-1]}")

    plt.legend()
    plt.title(r"$x$ Displacement of Different Swimmers over Time")
    plt.xlabel("Time (s)")
    plt.ylabel(r"$x$ Displacement (μm)")
    plt.show()


def plot_y_dist(swimmers, swimmer_creation, total_ticks):
    for swimmer_config, swimmer in zip(swimmer_creation, swimmers):
        time_space = np.linspace(0, total_ticks - 1, total_ticks) / swimmer.rate
        plt.plot(time_space, swimmer.y_dist, label=swimmer_config['label'])
        print(f"{swimmer_config['label']}: {swimmer.x_dist[-1]}")

    plt.legend()
    plt.title(r"$y$ Displacement of Different Swimmers over Time")
    plt.xlabel("Time (s)")
    plt.ylabel(r"$y$ Displacement (μm)")
    plt.show()


def plot_final_distances(swimmers, swimmer_creation):
    plt.plot(range(len(swimmers)), [max(swimmer.x_dist) for swimmer in swimmers], marker='o')
    plt.title("Displacements of Different Swimmers")
    plt.xlabel("Swimmer")
    plt.ylabel(r"$x$ Displacement (μm)")
    plt.xticks(range(len(swimmers)), [swimmer_config['label'] for swimmer_config in swimmer_creation], rotation=0)
    plt.show()


def plot_2d_displacements(swimmers, swimmer_creation):
    for swimmer, swimmer_config in zip(swimmers, swimmer_creation):
        plt.plot(swimmer.x_dist, swimmer.y_dist, label=swimmer_config['label'])
    plt.xlabel(r"$x$ Displacement (μm)")
    plt.ylabel(r"$y$ Displacement (μm)")
    plt.show()


def plot_2d_alphas(swimmers, swimmer_creation):
    for swimmer, swimmer_config in zip(swimmers, swimmer_creation):
        alphas = np.rollaxis(swimmer.alphas, 1)
        number_of_alphas = len(alphas)
        for i in range(number_of_alphas - 1):
            plt.plot(alphas[i], alphas[i + 1])
            plt.title(f"Phase Portrait for the {swimmer_config['label']}")
            plt.xlabel(r"$\alpha$" + str(i + 2).translate(SUB) + " (rad)", fontsize=15)
            plt.ylabel(r"$\alpha$" + str(i + 3).translate(SUB) + " (rad)", fontsize=15)
            plt.show()


def plot_3d_alphas(swimmers, swimmer_creation):
    for swimmer, swimmer_config in zip(swimmers, swimmer_creation):
        alphas = np.rollaxis(swimmer.alphas, 1)
        number_of_alphas = len(alphas)
        if number_of_alphas > 2:
            for i in range(number_of_alphas - 2):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(xs=alphas[i], ys=alphas[i + 1], zs=alphas[i + 2])
                ax.set_xlabel(r"$\alpha$" + str(i + 2).translate(SUB) + " (rad)")
                ax.set_ylabel(r"$\alpha$" + str(i + 3).translate(SUB) + " (rad)")
                ax.set_zlabel(r"$\alpha$" + str(i + 4).translate(SUB) + " (rad)")
                plt.title(f"3D Phase Portrait for the {swimmer_config['label']}")
                plt.show()


def plot_all_alphas(swimmers, swimmer_creation, total_ticks):
    for swimmer, swimmer_config in zip(swimmers, swimmer_creation):
        time_space = np.linspace(0, total_ticks - 1, total_ticks) / swimmer.rate
        alphas = np.rollaxis(swimmer.alphas, 1)
        for i in range(swimmer.N - 1):
            plt.plot(time_space, alphas[i], label=r"$\alpha$" + str(i + 2).translate(SUB))
        plt.legend()
        plt.title(r"$\alpha$'s over Time for" + f"{swimmer_config['label']}")
        plt.xlabel("Time (s)")
        plt.ylabel(r"$\alpha$ (rad)")
        plt.show()


def plot_positions(swimmers, number_of_overlays):
    for swimmer in swimmers:
        col_change = [0, 0, 1]
        values = [int(i/number_of_overlays*len(swimmer.coords_list)) for i in range(number_of_overlays + 1)]
        values[-1] = values[-1] - 1
        for value in values:
            xs = np.array([swimmer.coords_list[value][0][i] for i in range(swimmer.N + 1)])
            ys = np.array([swimmer.coords_list[value][1][i] for i in range(swimmer.N + 1)])
            plt.plot(xs, ys, color=tuple(col_change))
            col_change[0] += 1 / (number_of_overlays + 2)
            col_change[2] += -1 / (number_of_overlays + 2)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)  # labels along the bottom edge are off
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.gca().set_aspect("equal")
        plt.show()


def plot_positions_spaced(swimmers, number_of_overlays):
    for swimmer in swimmers:
        col_change = [0, 0, 1]
        values = [int(i / number_of_overlays * len(swimmer.coords_list)) for i in range(number_of_overlays + 1)]
        values[-1] = values[-1] - 1
        for i, value in enumerate(values):
            xs = np.array([swimmer.coords_list[value][0][i] for i in range(swimmer.N + 1)])
            ys = np.array([swimmer.coords_list[value][1][i] for i in range(swimmer.N + 1)])
            xs += space_between_positions * i
            plt.ylim([min(ys) - 80, max(ys) + 80])
            plt.plot(xs, ys, color=tuple(col_change))
            col_change[0] += 1 / (number_of_overlays + 2)
            col_change[2] += -1 / (number_of_overlays + 2)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)  # labels along the bottom edge are off
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.gca().set_aspect("equal")
        plt.show()