import numpy as np
import random


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
            is_excess = True if i == (length - 1) else a1[i] == a1[i+1]
            if is_excess and disjoint == 0:
                excess = excess + 1
            else:
                disjoint = disjoint + 1
        else:
            break

    return excess, disjoint


def species_distance(g1, g2, population_count, c1=1, c2=1, c3=1):
    excess, disjoint = calculate_excess_disjoint(g1, g2)
    (c1 * excess / population_count) + (c1 * disjoint / population_count) + (c3 + average_weights)


def evolve(population):
    # if did improve during last 15
    # breed top 40%
    # copy champion of each species with > 5 genes
    # 0.1% chance to mate with other species
    # 80% offspring mutation
    # 90% chance update weight & 10% to reset
    # 0.03 chance of new node for small population
    # 0.05 to add new link or 0.3 if population is big
    pass


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


class Population:
    def __init__(self, size, inputs, outputs):
        self.population = []
        self.node_innovation = inputs + outputs + 1
        self.conn_innovation = 1
        for _ in range(size):
            self.population.append(Genome(inputs, outputs))


class Genome:
    def __init__(self, input_count, output_count):
        self.nodes = []
        self.connections = []

        innovation = 1
        for _ in range(input_count):
            self.create_node(Node.TYPE_INPUT, innovation)
            innovation = innovation + 1

        for _ in range(output_count):
            self.create_node(Node.TYPE_OUTPUT, innovation)
            innovation = innovation + 1

    def create_node(self, node_type, innovation):
        new_node = Node(innovation, node_type)
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
        self.connections.append(first_node.connect_to(second_node, conn_innovation))

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


class Node:
    TYPE_INPUT = 1
    TYPE_OUTPUT = 2
    TYPE_HIDDEN = 3

    def __init__(self, innovation, node_type):
        self.innovation = innovation
        self.node_type = node_type
        self.in_connections = []
        self.out_connections = []

    def get_innovation(self):
        return self.innovation

    def get_type(self):
        return self.node_type

    def connect_to(self, next_node, conn_innovation):
        # create connection
        connection = Connections(conn_innovation)
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


class Connections:
    def __init__(self, innovation):
        self.in_node = None
        self.out_node = None
        self.innovation = innovation

    def get_next_node(self):
        return self.out_node

    def get_prev_node(self):
        return self.in_node

    def get_innovation(self):
        return self.innovation
