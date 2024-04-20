#                                 #          In the Name of GOD   # #
#
import numpy as np

class LinkPrediction:
    
    def __init__(self, multilayer_network):
        self.network = multilayer_network
    
    
    def common_neighbors(self, node1, node2):
        """
        Calculate the number of common neighbors between two nodes in a specific layer.
        """
        neighbors1 = set(self.network.node_map[node1]['edges']['node'])
        neighbors2 = set(self.network.node_map[node2]['edges']['node'])
        return len(neighbors1.intersection(neighbors2))
    
    
    def jaccard_coefficient(self, node1, node2):
        """
        Calculate the Jaccard coefficient between two nodes.
        """
        neighbors1 = set(self.network.node_map[node1]['edges']['node'])
        neighbors2 = set(self.network.node_map[node2]['edges']['node'])
        intersection = len(neighbors1.intersection(neighbors2))
        union = len(neighbors1.union(neighbors2))
        return intersection / union if union != 0 else 0
    
    
    def adamic_adar_index(self, node1, node2):
        """
        Calculate the Adamic-Adar index between two nodes, which considers the log of the degree of common neighbors.
        """
        neighbors1 = set(self.network.node_map[node1]['edges']['node'])
        neighbors2 = set(self.network.node_map[node2]['edges']['node'])
        common_neighbors = neighbors1.intersection(neighbors2)
        return sum(1 / np.log(len(self.network.node_map[n]['edges']['node'])) for n in common_neighbors if len(self.network.node_map[n]['edges']['node']) > 1)
    
    
    def predict_links(self, top_k=10, method='jaccard'):
        """
        Predict links for a given layer using a specified method.
        """
        scores = {}
        nodes = list(self.network.node_set)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1, node2 = nodes[i], nodes[j]
                if node1 in self.network.node_map[node2]['edges']['node']:
                    continue  # Skip if an edge already exists
                if method == 'jaccard':
                    score = self.jaccard_coefficient(node1, node2)
                elif method == 'common_neighbors':
                    score = self.common_neighbors(node1, node2)
                elif method == 'adamic_adar':
                    score = self.adamic_adar_index(node1, node2)
                scores[(node1, node2)] = score

        # Return the top k pairs with the highest scores
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

#end#
