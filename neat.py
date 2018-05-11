class Genome:
    def __init__(self, input_count, output_count):
        self.nodes = []
        self.connections = []
        self.id_counter = 1

        for _ in range(input_count):
            self.create_node(Node.TYPE_INPUT)

        for _ in range(output_count):
            self.create_node(Node.TYPE_OUTPUT)

    def create_node(self, node_type):
        new_node = Node(self.id_counter, node_type)
        self.nodes.append(new_node)
        self.id_counter = self.id_counter + 1
        return new_node

    def remove_node(self, node_id):
        for node in self.nodes:
            if node_id == node.get_id():
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

    def create_node_between(self, first, second):
        first_node = None
        second_node = None

        for n in self.nodes:
            if n.get_id() == first:
                first_node = n
                break

        for n in self.nodes:
            if n.get_id() == second:
                second_node = n
                break

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
        new_node = self.create_node(Node.TYPE_HIDDEN)

        # connect node
        self.connections.append(first_node.connect_to(new_node))
        self.connections.append(new_node.connect_to(second_node))


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
        return connection

    def get_next_connections(self):
        return self.out_connections

    def get_prev_connections(self):
        return self.in_connections


class Connections:
    def __init__(self):
        self.in_node = None
        self.out_node = None

    def get_next_node(self):
        return self.out_node

    def get_prev_node(self):
        return self.in_node
