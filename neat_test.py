import unittest
from neat import Node
from neat import Genome


class TestNodeMethods(unittest.TestCase):

    def test_connect_to(self):
        n1 = Node(1, Node.TYPE_INPUT)
        n2 = Node(2, Node.TYPE_OUTPUT)
        n1.connect_to(n2)

        self.assertEqual(2, n1.get_next_connections()[0].out_node.number)


class TestConnectionMethods(unittest.TestCase):

    def test_next(self):
        n1 = Node(1, Node.TYPE_INPUT)
        n2 = Node(2, Node.TYPE_OUTPUT)
        n1.connect_to(n2)

        self.assertEqual(2, n1.get_next_connections()[0].get_next().number)

    def test_prev(self):
        n1 = Node(1, Node.TYPE_INPUT)
        n2 = Node(2, Node.TYPE_OUTPUT)
        n1.connect_to(n2)

        self.assertEqual(1, n2.get_prev_connections()[0].get_prev().number)


class TestGenomeMethods(unittest.TestCase):

    def test_create_node(self):
        g = Genome(2, 1)
        self.assertEqual(1, g.nodes[0].get_id())
        self.assertEqual(2, g.nodes[1].get_id())
        self.assertEqual(3, g.nodes[2].get_id())

        self.assertEqual(Node.TYPE_INPUT, g.nodes[0].get_type())
        self.assertEqual(Node.TYPE_INPUT, g.nodes[1].get_type())
        self.assertEqual(Node.TYPE_OUTPUT, g.nodes[2].get_type())


if __name__ == '__main__':
    unittest.main()
