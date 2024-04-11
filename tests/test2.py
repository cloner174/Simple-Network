import unittest
from simpleN import MultilayerNetwork

class TestMultilayerNetwork(unittest.TestCase):
    def setUp(self):
        self.network = MultilayerNetwork()
    def test_add_layer(self):
        self.network.add_layer("Layer1")
        self.assertIn("Layer1", self.network.layers)
    def test_add_node(self):
        self.network.add_layer("Layer1")
        self.network.add_node("Layer1", "Node1")
        self.assertIn("Node1", self.network.nodes["Layer1"])
        self.assertIn("Node1", self.network.node_set)
    def test_add_edge_within_layer(self):
        self.network.add_layer("Layer1")
        self.network.add_node("Layer1", "Node1")
        self.network.add_node("Layer1", "Node2")
        self.network.add_edge("Node1", "Node2", "Layer1", 5)
        node1_index = self.network.nodes["Layer1"].index("Node1")
        node2_index = self.network.nodes["Layer1"].index("Node2")
        self.assertEqual(self.network.edges["Layer1"][node1_index, node2_index], 5)
    def test_add_inter_layer_edge(self):
        self.network.add_layer("Layer1")
        self.network.add_layer("Layer2")
        self.network.add_node("Layer1", "Node1")
        self.network.add_node("Layer2", "Node2")
        self.network.add_edge("Node1", "Node2", "Layer1", 5, "Layer2")
        self.assertIn((("Node1", "Layer1"), ("Node2", "Layer2"), 5), self.network.inter_layer_edges)
    def test_set_node_attribute(self):
        self.network.add_layer("Layer1")
        self.network.add_node("Layer1", "Node1")
        self.network.set_node_attribute("Node1", "color", "red")
        self.assertEqual(self.network.node_attributes["Node1"]["color"], "red")
    def test_set_edge_attribute(self):
        self.network.add_layer("Layer1")
        self.network.add_node("Layer1", "Node1")
        self.network.add_node("Layer1", "Node2")
        self.network.add_edge("Node1", "Node2", "Layer1", 5)
        self.network.set_edge_attribute("Node1", "Node2", "Layer1", "weight", 10)
        edge_key = ("Node1", "Node2", "Layer1") if self.network.directed else ("Node1", "Node2", "Layer1")
        self.assertEqual(self.network.edge_attributes[edge_key]["weight"], 10)

if __name__ == "__main__":
    unittest.main()
