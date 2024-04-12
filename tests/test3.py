import unittest
from simpleN import MultilayerNetwork, MNAnalysis
import scipy.sparse as sp
import numpy as np

class TestMNAnalysis(unittest.TestCase):
    
    def setUp(self):
        self.network = MultilayerNetwork(large_graph=True)
        self.network.add_layer("Layer1")
        for node in [1, 2, 3]:
            self.network.add_node("Layer1", node)
        for start, end in [(1, 2), (2, 3), (3, 1), (2, 1), (3, 2), (1, 3)]:
            self.network.add_edge(node1= start, layer_name1= "Layer1", node2 = end, layer_name2="Layer1",weight= 1)
        
        # Initialize the MNAnalysis with the setup network
        self.analysis = MNAnalysis(self.network)

    def test_community_detection(self):
        # Test community detection with a known outcome
        labels = self.analysis.detect_communities("Layer1", 2)
        # Since the network is a complete triangle, it's non-trivial how communities might be assigned in a split of 2
        # But we can check if the output length matches the number of nodes
        self.assertEqual(len(labels), 3)

    def test_calculate_global_efficiency(self):
        # Test global efficiency calculation
        efficiency = self.analysis.calculate_global_efficiency("Layer1")
        # As each node is reachable from each other in 1 step, efficiency should be 1
        self.assertEqual(efficiency, 1.0)

    def test_parallel_betweenness_centrality(self):
        # Test betweenness centrality calculation
        centrality_scores = self.analysis.parallel_betweenness_centrality("Layer1")
        # In a complete graph of three nodes, betweenness centrality for all should be 0
        self.assertTrue(all(score == 0 for score in centrality_scores))

    def test_aggregate_network(self):
        # Test the network aggregation functionality
        aggregated_matrix = self.analysis.aggregate_network()
        # As there's only one layer, the aggregated matrix should be the same as the layer's matrix
        expected_matrix = sp.csr_matrix(np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ]))
        self.assertTrue((aggregated_matrix != expected_matrix).nnz == 0)  # No non-zero difference elements

if __name__ == "__main__":
    unittest.main()
