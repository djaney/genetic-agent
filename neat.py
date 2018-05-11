class Genome:
    def __init__(self):
        self.nodes = []
        self.connections = []

    def mutate_nodes(self):
        pass

    def mutate_connections(self):
        pass


class Node:
    TYPE_INPUT = 1
    TYPE_OUTPUT = 2
    TYPE_HIDDEN = 3

    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.in_connections = []
        self.out_connections = []

    def connect_to(self, next_node):
        # create connection
        connection = Connections()
        # attach to self
        self.out_connections.append(connection)
        # attach to next
        next_node.in_connections.append(connection)
        # set connection codes
        connection.node_in = self
        connection.node_out = next_node


class Connections:
    def __init__(self):
        self.node_in = None
        self.node_out = None
