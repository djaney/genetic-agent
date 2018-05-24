import numpy as np
import math
import random
import matplotlib.pyplot as plt
from functools import reduce


def align_genome(g1, g2):
    max_innovation = np.max([g1.max_connection_innovation(), g2.max_connection_innovation()])
    a1 = []
    a2 = []
    for i in range(max_innovation):
        a1.append(g1.select_connection_by_innovation(i + 1, throw_not_found=False))
        a2.append(g2.select_connection_by_innovation(i + 1, throw_not_found=False))

    return a1, a2


def calculate_excess_disjoint(g1, g2):
    a1, a2 = align_genome(g1, g2)
    length = len(a1)
    excess = 0
    disjoint = 0
    for i in range(length - 1, 0, -1):
        if (a1[i] is None) != (a2[i] is None):
            # is excess
            is_excess = True if i == (length - 1) else a1[i] == a1[i + 1]
            if is_excess and disjoint == 0:
                excess = excess + 1
            else:
                disjoint = disjoint + 1
        else:
            break

    return excess, disjoint


def calculate_average_weights(gene):
    return np.mean([n.bias for n in gene.nodes] + [c.weight for c in gene.connections])


def species_distance(g1, g2, c1=1.0, c2=1.0, c3=3.0):
    excess, disjoint = calculate_excess_disjoint(g1, g2)
    average_weights = np.mean([calculate_average_weights(g1) + calculate_average_weights(g2)])
    g1_count = len(g1.connections)
    g2_count = len(g2.connections)
    max_genome_count = np.max([g1_count, g2_count])
    max_genome_count = 1 if max_genome_count < 20 else max_genome_count
    dist = (c1 * excess / max_genome_count) + (c2 * disjoint / max_genome_count) + (c3 * average_weights)
    return dist


def evolve(pool, other_species, generation, elite_size=0.4, champ_threshold=5, history_check=15, mutation=0.8,
           weight_update=0.9, new_node=0.03, new_link=0.05, cross_breed=0.001):
    population_size = len(pool)
    new_population = []

    # if did improve during last 15
    last_hist_list = [g.score_history[-history_check] for g in pool if len(g.score_history) >= history_check]
    last_score = np.max([g.score for g in pool])
    last_hist_mean = np.mean(last_hist_list) if len(last_hist_list) > 0 else None
    did_improved = last_hist_mean is None or last_hist_mean < last_score
    if did_improved:
        # breed top 40%
        pool = sorted(pool, key=lambda x: x.score, reverse=True)
        elite = pool[:math.floor(population_size * elite_size)]
        new_population = new_population + breed(elite, generation, mutation=mutation, weight_update=weight_update,
                                                new_node=new_node, new_link=new_link)
        # copy champion of each species with minimum size
        if population_size > champ_threshold:
            new_population.append(pool[0])

        # chance to mate with other species
        if random.random() < cross_breed:
            new_population.append(breed(random.choice(elite), random.choice(other_species)))

        new_population = new_population + pool[population_size - len(new_population):]

    else:
        new_population = pool

    return new_population


def breed(population, generation, mutation, weight_update, new_node, new_link):
    initial_size = len(population)
    new_population = []
    for _ in range(initial_size):
        sample = random.sample(population, 2)
        a1, a2 = align_genome(sample[0], sample[1])
        g = crossover(a1, a2)
        g.generation = generation
        new_population.append(g)
    # 80% offspring mutation
    if random.random() < mutation:
        # 90% chance update weight & 10% to reset TODO
        # 0.03 chance of new node for small population TODO
        pass

    return new_population


def crossover(a1, a2):
    if len(a1) != len(a2):
        raise Exception('inputs not the same length')

    child_connections = []
    for i in range(len(a1)):

        if a1[i] is not None and a2[i] is not None:
            if random.random() > 0.5:
                child_connections.append(a1[i])
            else:
                child_connections.append(a2[i])
        elif a1[i] is None and a2[i] is not None:
            child_connections.append(a2[i])
        elif a1[i] is not None and a2[i] is None:
            child_connections.append(a1[i])
        else:
            # disjoint
            pass
    child_genome = Genome(0, 0)
    for c in child_connections:
        prev_node = c.get_prev_node()
        next_node = c.get_next_node()
        if c not in child_genome.connections:
            child_genome.connections.append(c)
        if not child_genome.has_node_with_innovation(prev_node.get_innovation()):
            child_genome.nodes.append(prev_node)
        if not child_genome.has_node_with_innovation(next_node.get_innovation()):
            child_genome.nodes.append(next_node)
    return child_genome


def speciate(population, existing_species, c1, c2, c3, species_distance_threshold):
    new_species = {}
    for i, g in enumerate(population):
        added = False
        for species in list(existing_species.keys()):
            dist = species_distance(g, existing_species[species][0], c1, c2, c3)
            if species_distance_threshold >= dist:
                new_species[species] = new_species.get(species, [])
                new_species[species].append(g)
                added = True
                break

        for species in list(new_species.keys()):
            dist = species_distance(g, new_species[species][0], c1, c2, c3)
            if species_distance_threshold >= dist:
                new_species[species] = new_species.get(species, [])
                new_species[species].append(g)
                added = True
                break

        if not added:
            name = 's' + str(len(new_species.keys()))
            new_species[name] = new_species.get(name, [])
            new_species[name].append(g)
    return new_species


class Population:
    def __init__(self, size, inputs, outputs, c1=1.0, c2=1.0, c3=3.0, species_distance_threshold=4.0):
        self.node_innovation = inputs + outputs + 1
        self.conn_innovation = 1
        self.population = {}
        population = []
        for _ in range(size):
            population.append(Genome(inputs, outputs))

        self.population = speciate(population, self.population, c1=c1, c2=c2, c3=c3,
                                   species_distance_threshold=species_distance_threshold)


class Genome:
    def __init__(self, input_count, output_count, initializer=None):
        self.nodes = []
        self.connections = []
        self.score_history = []
        self.score = 0
        self.generation = 1
        self.initializer = initializer

        innovation = 1
        for _ in range(input_count):
            self.create_node(Node.TYPE_INPUT, innovation)
            innovation = innovation + 1

        for _ in range(output_count):
            self.create_node(Node.TYPE_OUTPUT, innovation)
            innovation = innovation + 1

    def set_score(self, score):
        self.score_history.append(self.score)
        self.score = score

    def create_node(self, node_type, innovation):
        new_node = Node(innovation, node_type, initializer=self.initializer)
        self.nodes.append(new_node)
        return new_node

    def remove_node(self, node_id):
        for node in self.nodes:
            if node_id == node.get_innovation():
                prev_connections = node.get_prev_connections()
                next_connections = node.get_next_connections()
                # remove node
                self.nodes.remove(node)

                # remove connection from previous node
                for c in prev_connections:
                    c.get_prev_node().get_next_connections().remove(c)

                # remove connections from next nodes
                for c in next_connections:
                    c.get_next_node().get_prev_connections().remove(c)
                break

    def create_node_between(self, first_id, second_id, innovation, conn_innovation, existing_connection=None):

        first_node = self.select_node_by_id(first_id)
        second_node = self.select_node_by_id(second_id)

        # check first node
        if first_node is None:
            raise Exception("first node doest not exist")

        # check second node
        if second_node is None:
            raise Exception("first node doest not exist")

        # check if connected
        # for c in first_node.get_next_connections():
        #     if c.get_next_node() is second_node:
        #         existing_connection = c

        if existing_connection is not None:
            # if there is an existing connection
            # remove the connection
            existing_connection.get_prev_node().get_next_connections().remove(existing_connection)
            existing_connection.get_next_node().get_prev_connections().remove(existing_connection)
            self.connections.remove(existing_connection)

        # create node
        new_node = self.create_node(Node.TYPE_HIDDEN, innovation)

        # connect node
        self.connect_nodes(first_node, new_node, conn_innovation)
        self.connect_nodes(new_node, second_node, conn_innovation + 1)

    def max_connection_innovation(self):
        max_innovation = 0
        for c in self.connections:
            max_innovation = np.max([c.get_innovation(), max_innovation])

        return max_innovation

    def connect_nodes(self, first_node, second_node, conn_innovation):
        # connect node
        self.connections.append(first_node.connect_to(second_node, conn_innovation, initializer=self.initializer))

    def connect_nodes_by_id(self, first_id, second_id, conn_innovation):
        first_node = self.select_node_by_id(first_id)
        second_node = self.select_node_by_id(second_id)
        # connect node
        self.connect_nodes(first_node, second_node, conn_innovation)

    def select_node_by_id(self, node_id):
        node = None
        for n in self.nodes:
            if n.get_innovation() == node_id:
                node = n
                break

        if node is None:
            raise Exception("node not found")

        return node

    def has_node_with_innovation(self, node_id):
        for n in self.nodes:
            if n.get_innovation() == node_id:
                return True

        return False

    def mutate_nodes(self):
        pass

    def mutate_connections(self):
        pass

    def select_connection_by_innovation(self, innovation, throw_not_found=True):
        connection = None
        for c in self.connections:
            if c.get_innovation() == innovation:
                connection = c
                break

        if throw_not_found and connection is None:
            raise Exception("connection not found")

        return connection

    def get_input_nodes(self):
        sorted(self.nodes, key=lambda n: n.get_innovation())
        return [n for n in self.nodes if n.get_type() == Node.TYPE_INPUT]

    def get_output_nodes(self):
        sorted(self.nodes, key=lambda n: n.get_innovation())
        return [n for n in self.nodes if n.get_type() == Node.TYPE_OUTPUT]

    def reset_values(self):
        for n in self.nodes:
            n.reset_value()

    def reset_activated(self):
        for n in self.nodes:
            n.reset_activated()

        for c in self.connections:
            c.reset_activated()

    def evaluate_layer(self, nodes):
        # calculate values for each nodes
        for n in nodes:
            value = 0
            # get each connection
            for c in n.get_prev_connections():
                # get node behind that connection
                if c.get_prev_node().get_value() is not None:
                    # value is sum if prev node value * weight
                    value = value + c.get_prev_node().get_value() * c.get_weight()
                    c.set_activated(True)
            # then add bias
            value = value + n.get_bias()
            n.set_activated(True)
            n.set_value(value)

        # evaluate next
        for n in nodes:
            next_nodes = n.get_next_nodes()
            next_nodes = [n for n in next_nodes if not n.get_activated()]
            if len(next_nodes) > 0:
                self.evaluate_layer(n.get_next_nodes())

    def is_nodes_activated(self):
        return reduce(lambda a, b: a + (0 if b.get_activated() else 1), self.nodes, 0) == 0

    def is_connections_activated(self):
        return reduce(lambda a, b: a + (0 if b.get_activated() else 1), self.connections, 0) == 0

    def is_all_activated(self):
        return self.is_nodes_activated() and self.is_connections_activated()

    def run(self, input):
        self.reset_values()
        input_nodes = self.get_input_nodes()

        if len(input) != len(input_nodes):
            raise Exception("input count must be the same as number of input nodes")

        # evaluate continuously while there are nodes without value
        while True:
            print('pass')
            self.reset_activated()
            # set value if inputs
            for k, n in enumerate(input_nodes):
                n.set_value(input[k])
            self.evaluate_layer(input_nodes)
            if self.is_all_activated():
                break
        return [n.get_value() for n in self.get_output_nodes()]


class Node:
    TYPE_INPUT = 1
    TYPE_OUTPUT = 2
    TYPE_HIDDEN = 3

    def __init__(self, innovation, node_type, initializer=None):
        self.innovation = innovation
        self.node_type = node_type
        self.bias = 0
        self.in_connections = []
        self.out_connections = []
        self.value = None
        self.activated = None

        if initializer is not None:
            self.bias = initializer()

    def reset_value(self):
        self.value = None

    def reset_activated(self):
        self.value = None

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def set_activated(self, activated):
        self.activated = activated

    def get_activated(self):
        return self.activated

    def set_bias(self, bias):
        self.bias = bias

    def get_bias(self):
        return self.bias

    def get_innovation(self):
        return self.innovation

    def get_type(self):
        return self.node_type

    def connect_to(self, next_node, conn_innovation, initializer=None):
        # create connection
        connection = Connections(conn_innovation, initializer=initializer)
        # attach to self
        self.out_connections.append(connection)
        # attach to next
        next_node.in_connections.append(connection)
        # set connection codes
        connection.in_node = self
        connection.out_node = next_node
        return connection

    def get_next_connections(self):
        return self.out_connections

    def get_prev_connections(self):
        return self.in_connections

    def is_connected_to_next_by_id(self, node_id):
        result = False
        for c in self.get_next_connections():
            if c.get_next_node().get_innovation() == node_id:
                result = True
        return result

    def is_connected_to_prev_by_id(self, node_id):
        result = False
        for c in self.get_prev_connections():
            if c.get_prev_node().get_innovation() == node_id:
                result = True
        return result

    def get_next_nodes(self):
        nodes = []
        for c in self.get_next_connections():
            nodes.append(c.get_next_node())
        return nodes


class Connections:
    def __init__(self, innovation, initializer=None):
        self.in_node = None
        self.out_node = None
        self.innovation = innovation
        self.weight = 0
        self.activated = False
        if initializer is not None:
            self.weight = initializer()

    def reset_activated(self):
        self.activated = False

    def set_activated(self, activated):
        self.activated = activated

    def get_activated(self):
        return self.activated

    def get_next_node(self):
        return self.out_node

    def get_prev_node(self):
        return self.in_node

    def get_innovation(self):
        return self.innovation

    def get_weight(self):
        return self.weight


class Printer:
    WIDTH = 5

    def __init__(self, genome):
        self.genome = genome
        self.finished_nodes = []
        self.layer = 0
        self.printed_nodes = {}
        fig, ax = plt.subplots(1)
        self.ax = ax
        self.scatter_x = []
        self.scatter_y = []

    def print(self):
        self.iterate_layer(self.genome.get_input_nodes())
        self.iterate_layer(self.genome.get_output_nodes())
        self.plot(self.genome)

    def iterate_layer(self, nodes):
        x = self.printed_nodes
        y = self.print_layer(nodes, layer=self.layer)
        self.printed_nodes = {**x, **y}
        next_nodes = []
        for n in nodes:
            for n2 in n.get_next_nodes():
                if n2 not in self.finished_nodes and n2.get_type() != Node.TYPE_OUTPUT:
                    self.finished_nodes.append(n2)
                    next_nodes.append(n2)

        self.layer = self.layer + 1
        if next_nodes:
            self.iterate_layer(next_nodes)

    def print_layer(self, nodes, layer=0):
        printed = {}

        node_count = len(nodes)
        distance = self.WIDTH / (node_count + 1)

        for k, n in enumerate(nodes):
            x = distance + (k * distance)
            y = layer
            self.scatter_x.append(x)
            self.scatter_y.append(y)
            printed[n.get_innovation()] = (x, y)
        return printed

    def plot(self, genome):
        # print connections
        for c in genome.connections:
            prev_innovation = c.get_prev_node().get_innovation()
            next_innovation = c.get_next_node().get_innovation()
            from_x, from_y = self.printed_nodes.get(prev_innovation)
            to_x, to_y = self.printed_nodes.get(next_innovation)
            self.ax.arrow(from_x, from_y, to_x - from_x, to_y - from_y, head_width=0.03, head_length=0.1, fc='k',
                          ec='k',
                          length_includes_head=True)

        # print dots
        self.ax.scatter(self.scatter_x, self.scatter_y, s=500)

        # hide axis ticks
        self.ax.set_yticklabels([])
        self.ax.set_xticklabels([])

        # show graph
        plt.show()
