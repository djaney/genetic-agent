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

    def create_node_between(self, first_id, second_id, innovation, conn_innovation):

        first_node = self.select_node_by_id(first_id)
        second_node = self.select_node_by_id(second_id)

        # check first node
        if first_node is None:
            raise Exception("first node doest not exist")

        # check second node
        if second_node is None:
            raise Exception("first node doest not exist")

        # check if connected
        existing_connection = None
        for c in first_node.get_next_connections():
            if c.get_next_node() is second_node:
                existing_connection = c

        if existing_connection is not None:
            # if there is an existing connection
            # remove the connection
            first_node.get_next_connections().remove(existing_connection)
            second_node.get_prev_connections().remove(existing_connection)
            self.connections.remove(existing_connection)

        # create node
        new_node = self.create_node(Node.TYPE_HIDDEN, innovation)

        # connect node
        self.connect_nodes(first_node, new_node, conn_innovation)
        self.connect_nodes(new_node, second_node, conn_innovation+1)

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

    def mutate_nodes(self):
        pass

    def mutate_connections(self):
        pass


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
