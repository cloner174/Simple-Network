import unittest
from simpleN import MultilayerNetwork 

class TestMultilayerNetwork(unittest.TestCase):
    
    def setUp(self):
        """Initialize a MultilayerNetwork instance before each test."""
        self.network = MultilayerNetwork()
    def test_add_layer(self):
        """Test adding layers to the network."""
        self.network.add_layer("Layer1")
        self.assertIn("Layer1", self.network.layers)
        self.assertIn("Layer1", self.network.nodes)
        self.assertIn("Layer1", self.network.edges)
    def test_add_node(self):
        """Test adding nodes to a layer."""
        self.network.add_layer("Layer1")
        self.network.add_node("Layer1", "Node1")
        self.assertIn("Node1", self.network.node_set)
        self.assertIn("Node1", self.network.nodes["Layer1"])
    def test_add_edge_within_layer(self):
        """Test adding an edge within a layer."""
        self.network.add_layer("Layer1")
        self.network.add_node("Layer1", "Node1")
        self.network.add_node("Layer1", "Node2")
        self.network.add_edge("Node1", "Node2", "Layer1", 1)
        if not self.network.large_graph:
            self.assertEqual(self.network.edges["Layer1"][0][1], 1)
    def test_add_inter_layer_edge(self):
        """Test adding an inter-layer edge."""
        self.network.add_layer("Layer1")
        self.network.add_layer("Layer2")
        self.network.add_node("Layer1", "Node1")
        self.network.add_node("Layer2", "Node2")
        self.network.add_inter_layer_edge("Node1", "Layer1", "Node2", "Layer2", 1)
        self.assertIn((("Node1", "Layer1"), ("Node2", "Layer2"), 1), self.network.inter_layer_edges)
    def test_add_edge_nonexistent_node(self):
        """Test adding an edge with a nonexistent node."""
        self.network.add_layer("Layer1")
        with self.assertRaises(ValueError):
            self.network.add_edge("Node1", "Node2", "Layer1", 1)
    def test_add_inter_layer_edge_nonexistent_layer(self):
        """Test adding an inter-layer edge with a nonexistent layer."""
        with self.assertRaises(ValueError):
            self.network.add_inter_layer_edge("Node1", "NonexistentLayer", "Node2", "Layer2", 1)

if __name__ == "__main__":
    unittest.main()