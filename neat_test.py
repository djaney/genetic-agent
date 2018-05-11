import unittest
from neat import Node
from neat import Genome


class TestNodeMethods(unittest.TestCase):

    def test_connect_to(self):
        n1 = Node(1, Node.TYPE_INPUT)
        n2 = Node(2, Node.TYPE_OUTPUT)
        n1.connect_to(n2)

        self.assertEqual(2, n1.get_next_connections()[0].out_node.get_id())


class TestConnectionMethods(unittest.TestCase):

    def test_next(self):
        n1 = Node(1, Node.TYPE_INPUT)
        n2 = Node(2, Node.TYPE_OUTPUT)
        n1.connect_to(n2)

        self.assertEqual(2, n1.get_next_connections()[0].get_next_node().get_id())

    def test_prev(self):
        n1 = Node(1, Node.TYPE_INPUT)
        n2 = Node(2, Node.TYPE_OUTPUT)
        n1.connect_to(n2)

        self.assertEqual(1, n2.get_prev_connections()[0].get_prev_node().get_id())


class TestGenomeMethods(unittest.TestCase):

    def test_init(self):
        g = Genome(2, 1)
        self.assertEqual(1, g.nodes[0].get_id())
        self.assertEqual(2, g.nodes[1].get_id())
        self.assertEqual(3, g.nodes[2].get_id())

        self.assertEqual(Node.TYPE_INPUT, g.nodes[0].get_type())
        self.assertEqual(Node.TYPE_INPUT, g.nodes[1].get_type())
        self.assertEqual(Node.TYPE_OUTPUT, g.nodes[2].get_type())

    def test_remove_node(self):
        g = Genome(1, 1)
        g.create_node(Node.TYPE_HIDDEN)
        g.nodes[0].connect_to(g.nodes[2])
        g.nodes[2].connect_to(g.nodes[1])

        self.assertEqual(3, g.nodes[0].get_next_connections()[0].get_next_node().get_id())
        self.assertEqual(2, g.nodes[0].get_next_connections()[0].get_next_node().get_next_connections()[
            0].get_next_node().get_id())

        g.remove_node(3)
        self.assertEqual(0, len(g.connections))
        self.assertEqual(0, len(g.nodes[0].get_next_connections()))
        self.assertEqual(0, len(g.nodes[1].get_prev_connections()))

    def test_create_node_between(self):
        g = Genome(1, 1)

        g.create_node_between(1, 2)
        self.assertEqual(2, len(g.connections))

        g.create_node_between(3, 2)

        self.assertEqual(3, len(g.connections))

        current_node = g.nodes[0]
        self.assertEqual(1, g.nodes[0].get_id())

        current_node = current_node.get_next_connections()[0].get_next_node()
        self.assertEqual(3, current_node.get_id())

        current_node = current_node.get_next_connections()[0].get_next_node()
        self.assertEqual(4, current_node.get_id())

        current_node = current_node.get_next_connections()[0].get_next_node()
        self.assertEqual(2, current_node.get_id())


if __name__ == '__main__':
    unittest.main()
