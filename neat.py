class Genome:
    def __init__(self, input_count, output_count):
        self.nodes = []
        self.connections = []
        self.id_counter = 1

        for _ in range(input_count):
            self.create_node(Node.TYPE_INPUT)

        for _ in range(input_count):
            self.create_node(Node.TYPE_OUTPUT)

    def create_node(self, node_type):
        self.nodes.append(Node(self.id_counter, node_type))
        self.id_counter = self.id_counter + 1

    def mutate_nodes(self):
        pass

    def mutate_connections(self):
        pass


class Node:
    TYPE_INPUT = 1
    TYPE_OUTPUT = 2
    TYPE_HIDDEN = 3

    def __init__(self, number, node_type):
        self.number = number
        self.node_type = node_type
        self.in_connections = []
        self.out_connections = []

    def get_id(self):
        return self.number

    def get_type(self):
        return self.node_type


    def connect_to(self, next_node):
        # create connection
        connection = Connections()
        # attach to self
        self.out_connections.append(connection)
        # attach to next
        next_node.in_connections.append(connection)
        # set connection codes
        connection.in_node = self
        connection.out_node = next_node

    def get_next_connections(self):
        return self.out_connections

    def get_prev_connections(self):
        return self.in_connections


class Connections:
    def __init__(self):
        self.in_node = None
        self.out_node = None

    def get_next(self):
        return self.out_node

    def get_prev(self):
        return self.in_node
