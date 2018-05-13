import unittest
from neat import Node, Genome, Population


class TestPopulationMethods(unittest.TestCase):
    def test_init(self):
        p = Population(10, 3, 1)
        self.assertEqual(10, len(p.population))
        self.assertEqual(5, p.node_innovation)
        self.assertEqual(1, p.conn_innovation)

    def test_align_genome(self):
        g1 = Genome(3, 1)
        g1.create_node(Node.TYPE_HIDDEN, 5)
        g1.connect_nodes_by_id(1, 4, 1)
        g1.connect_nodes_by_id(2, 4, 2)
        g1.connect_nodes_by_id(3, 4, 3)
        g1.connect_nodes_by_id(2, 5, 4)
        g1.connect_nodes_by_id(5, 4, 5)
        g1.connect_nodes_by_id(1, 5, 6)

        g2 = Genome(3, 1)
        g2.create_node(Node.TYPE_HIDDEN, 5)
        g2.create_node(Node.TYPE_HIDDEN, 6)

        g2.connect_nodes_by_id(1, 4, 1)
        g2.connect_nodes_by_id(2, 4, 2)
        g2.connect_nodes_by_id(3, 4, 3)
        g2.connect_nodes_by_id(2, 5, 4)
        g2.connect_nodes_by_id(5, 4, 5)
        g2.connect_nodes_by_id(5, 6, 6)
        g2.connect_nodes_by_id(6, 4, 7)
        g2.connect_nodes_by_id(3, 5, 9)
        g2.connect_nodes_by_id(1, 6, 10)


class TestNodeMethods(unittest.TestCase):

    def test_connect_to(self):
        n1 = Node(1, Node.TYPE_INPUT)
        n2 = Node(2, Node.TYPE_OUTPUT)
        n1.connect_to(n2, 1)

        self.assertEqual(2, n1.get_next_connections()[0].out_node.get_innovation())


class TestConnectionMethods(unittest.TestCase):

    def test_next(self):
        n1 = Node(1, Node.TYPE_INPUT)
        n2 = Node(2, Node.TYPE_OUTPUT)
        n1.connect_to(n2, 1)

        self.assertEqual(2, n1.get_next_connections()[0].get_next_node().get_innovation())
        self.assertEqual(1, n1.get_next_connections()[0].get_innovation())

    def test_prev(self):
        n1 = Node(1, Node.TYPE_INPUT)
        n2 = Node(2, Node.TYPE_OUTPUT)
        n1.connect_to(n2, 1)

        self.assertEqual(1, n2.get_prev_connections()[0].get_prev_node().get_innovation())


class TestGenomeMethods(unittest.TestCase):

    def test_init(self):
        g = Genome(2, 1)
        self.assertEqual(1, g.nodes[0].get_innovation())
        self.assertEqual(2, g.nodes[1].get_innovation())
        self.assertEqual(3, g.nodes[2].get_innovation())

        self.assertEqual(Node.TYPE_INPUT, g.nodes[0].get_type())
        self.assertEqual(Node.TYPE_INPUT, g.nodes[1].get_type())
        self.assertEqual(Node.TYPE_OUTPUT, g.nodes[2].get_type())

    def test_remove_node(self):
        g = Genome(1, 1)
        g.create_node(Node.TYPE_HIDDEN, 3)
        g.nodes[0].connect_to(g.nodes[2], 1)
        g.nodes[2].connect_to(g.nodes[1], 2)

        self.assertEqual(3, g.nodes[0].get_next_connections()[0].get_next_node().get_innovation())
        self.assertEqual(2, g.nodes[0].get_next_connections()[0].get_next_node().get_next_connections()[
            0].get_next_node().get_innovation())

        g.remove_node(3)
        self.assertEqual(0, len(g.connections))
        self.assertEqual(0, len(g.nodes[0].get_next_connections()))
        self.assertEqual(0, len(g.nodes[1].get_prev_connections()))

    def test_create_node_between(self):
        g = Genome(1, 1)

        g.create_node_between(1, 2, 3, 1)
        self.assertEqual(2, len(g.connections))
        self.assertEqual(1, g.connections[0].get_innovation())
        self.assertEqual(2, g.connections[1].get_innovation())

        g.create_node_between(3, 2, 4, 3, g.select_connection_by_innovation(2))

        self.assertEqual(3, len(g.connections))

        current_node = g.nodes[0]
        self.assertEqual(1, g.nodes[0].get_innovation())

        current_node = current_node.get_next_connections()[0].get_next_node()
        self.assertEqual(3, current_node.get_innovation())

        current_node = current_node.get_next_connections()[0].get_next_node()
        self.assertEqual(4, current_node.get_innovation())

        current_node = current_node.get_next_connections()[0].get_next_node()
        self.assertEqual(2, current_node.get_innovation())

    def test_mutate_add_connection(self):
        g = Genome(3, 1)
        g.connect_nodes_by_id(1, 4, 1)
        g.connect_nodes_by_id(2, 4, 2)
        g.connect_nodes_by_id(3, 4, 3)
        g.create_node_between(2, 4, 5, 4)
        g.connect_nodes_by_id(1, 5, 6)

        self.assertEqual(True, g.select_node_by_id(1).is_connected_to_next_by_id(4))
        self.assertEqual(True, g.select_node_by_id(4).is_connected_to_prev_by_id(1))
        self.assertEqual(True, g.select_node_by_id(1).is_connected_to_next_by_id(5))
        self.assertEqual(True, g.select_node_by_id(2).is_connected_to_next_by_id(5))
        self.assertEqual(True, g.select_node_by_id(5).is_connected_to_next_by_id(4))
        self.assertEqual(True, g.select_node_by_id(3).is_connected_to_next_by_id(4))

        g.connect_nodes_by_id(3, 5, 7)
        self.assertEqual(True, g.select_node_by_id(3).is_connected_to_next_by_id(5))

    def test_mutate_add_node(self):
        g = Genome(3, 1)
        g.connect_nodes_by_id(1, 4, 1)
        g.connect_nodes_by_id(2, 4, 2)
        g.connect_nodes_by_id(3, 4, 3)
        g.create_node_between(2, 4, 5, 4)
        g.connect_nodes_by_id(1, 5, 6)

        self.assertEqual(True, g.select_node_by_id(1).is_connected_to_next_by_id(4))
        self.assertEqual(True, g.select_node_by_id(4).is_connected_to_prev_by_id(1))
        self.assertEqual(True, g.select_node_by_id(1).is_connected_to_next_by_id(5))
        self.assertEqual(True, g.select_node_by_id(2).is_connected_to_next_by_id(5))
        self.assertEqual(True, g.select_node_by_id(5).is_connected_to_next_by_id(4))
        self.assertEqual(True, g.select_node_by_id(3).is_connected_to_next_by_id(4))

        g.create_node_between(3, 4, 6, 7)
        self.assertIsNotNone(g.select_node_by_id(6))
        self.assertRaises(Exception, lambda: g.select_node_by_id(7))


if __name__ == '__main__':
    unittest.main()
