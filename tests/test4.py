import unittest
from simpleN import MultilayerNetwork, MNAnalysis


class TestMNAnalysis(unittest.TestCase):
    
    def setUp(self):
        self.network = MultilayerNetwork()
        self.network.add_layer('Layer1')
        self.network.add_node('Layer1', 'Node1')
        self.network.add_node('Layer1', 'Node2')
        self.network.add_edge('Node1', 'Node2', layer_name1='Layer1', layer_name2='Layer1', weight=1)
        self.analysis = MNAnalysis(self.network)
    
    def test_layerwise_degree_distribution(self):
        distribution = self.analysis.layerwise_degree_distribution()
        self.assertIn('Layer1', distribution)
        self.assertEqual(len(distribution['Layer1']), 2) 
        self.assertTrue(any(score > 0 for score in distribution['Layer1']))
    
    def test_detect_communities(self):
        self.network.add_node('Layer1', 'Node3')
        self.network.add_edge('Node1', 'Node3', layer_name1='Layer1', layer_name2='Layer1', weight=1)
        self.network.add_edge('Node2', 'Node3', layer_name1='Layer1', layer_name2='Layer1', weight=1)
        communities = self.analysis.detect_communities('Layer1', n_clusters=2)
        self.assertEqual(len(communities), 3)
        self.assertTrue(max(communities) < 2)
    
    def test_global_efficiency(self):
        efficiency = self.analysis.calculate_global_efficiency('Layer1')
        self.assertIsInstance(efficiency, float)
        self.assertGreaterEqual(efficiency, 0)
        self.assertLessEqual(efficiency, 1)
    
    def test_count_connected_components(self):
        self.network.add_layer('Layer2')
        self.network.add_node('Layer2', 'Node1')
        self.network.add_node('Layer2', 'Node2')
        self.network.add_edge(node1='Node1', node2='Node2', layer_name1='Layer2', weight=1)
        components_layer1 = self.analysis.count_connected_components('Layer1')
        components_layer2 = self.analysis.count_connected_components('Layer2')
        self.assertEqual(components_layer1, 1)
        self.assertEqual(components_layer2, 1)

if __name__ == '__main__':
    unittest.main()