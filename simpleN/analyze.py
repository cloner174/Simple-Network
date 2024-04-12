#                                 #          In the Name of GOD   # #
#
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path, dijkstra
from scipy.sparse.linalg import eigs
from sklearn.cluster import SpectralClustering
from joblib import Parallel, delayed


class MNAnalysis:
    
    def __init__(self, multilayer_network):
        """
        Initialize the analysis class with a MultilayerNetwork instance.
        """
        self.network = multilayer_network
    
    
    def layerwise_degree_distribution(self):
        """
        Calculate the degree distribution for each layer of the network.
        """
        degree_distributions = {}
        for layer in self.network.layers:
            degrees = self.network.calculate_layer_degrees(layer)
            degree_distributions[layer] = np.bincount(degrees) / float(len(degrees))
        return degree_distributions
    
    
    def aggregate_network(self):
        """
        Aggregate the multilayer network into a single-layer network.
        This method combines all layers into one, summing up the weights of inter-layer edges.
        """
        aggregated_matrix = None
        for layer in self.network.layers:
            matrix = self.network.edges[layer]
            if isinstance(matrix, sp.lil_matrix):
                matrix = matrix.tocsr()
            if aggregated_matrix is None:
                aggregated_matrix = matrix
            else:
                aggregated_matrix += matrix
        return aggregated_matrix
    
    
    def detect_communities(self, layer_name, n_clusters=2):
        """
        Detect communities within a specific layer using spectral clustering.
        """
        if layer_name not in self.network.layers:
            raise ValueError(f"Layer {layer_name} not found.")
        
        adjacency_matrix = self.network.edges[layer_name]
        if not sp.issparse(adjacency_matrix):
            adjacency_matrix = sp.csr_matrix(adjacency_matrix)
        
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        labels = clustering.fit_predict(adjacency_matrix)
        
        return labels
    
    
    def calculate_global_efficiency(self, layer_name):
        matrix = self.network.edges[layer_name]
        if not sp.issparse(matrix):
            matrix = sp.csr_matrix(matrix)
        
        distances = dijkstra(matrix, directed=self.network.directed, unweighted=True)
        finite_distances = distances[np.isfinite(distances) & (distances > 0)]
        
        if finite_distances.size == 0:
            return 0  # Return 0 efficiency if there are no valid paths
        
        efficiency = np.mean(1. / finite_distances)
        return efficiency
    
    
    def count_connected_components(self, layer_name):
        """
        Count the number of connected components in a specific layer.
        """
        if layer_name not in self.network.layers:
            raise ValueError(f"Layer {layer_name} not found.")
        
        matrix = self.network.edges[layer_name]
        if isinstance(matrix, list):
            matrix = np.array(matrix)
        if not sp.issparse(matrix):
            matrix = sp.csr_matrix(matrix)
        
        n_components, _ = connected_components(csgraph=matrix, directed=self.network.directed, return_labels=True)
        return n_components
    
    
    def analyze_dynamic_changes(self, snapshots):
        """
        Analyze dynamic changes in the network over a series of snapshots. Each snapshot is a MultilayerNetwork instance.
        Returns a list of changes in global efficiency over time.
        """
        efficiencies = []
        for snapshot in snapshots:
            efficiency_per_layer = {}
            for layer_name in snapshot.layers:
                efficiency = self.calculate_global_efficiency(layer_name)
                efficiency_per_layer[layer_name] = efficiency
            efficiencies.append(efficiency_per_layer)
        return efficiencies
    
    
    def explore_inter_layer_connectivity(self):
        """
        Explore and quantify the connectivity patterns between layers.
        This method calculates the density of inter-layer edges and the distribution of weights.
        """
        inter_layer_edges = self.network.get_inter_layer_edges()
        total_inter_layer_edges = len(inter_layer_edges)
        if total_inter_layer_edges == 0:
            return {'density': 0, 'weight_distribution': []}
        
        total_possible_inter_layer_edges = sum(len(self.network.nodes[layer]) for layer in self.network.layers) ** 2
        density = total_inter_layer_edges / total_possible_inter_layer_edges
        
        weights = [weight for _, _, _, _, weight in inter_layer_edges]
        weight_distribution = np.histogram(weights, bins=10, density=True)[0]
        
        return {'density': density, 'weight_distribution': weight_distribution.tolist()}
    
    
    def parallel_betweenness_centrality(self, layer_name):
        matrix = self.network.edges[layer_name]
        if not isinstance(matrix, sp.csr_matrix):
            matrix = sp.csr_matrix(matrix)
        n = matrix.shape[0]
        
        def compute_for_node(start):
            _, predecessors = shortest_path(csgraph=matrix, directed=self.network.directed, indices=start, return_predecessors=True)
            betweenness = np.zeros(n)
            
            for end in range(n):
                if start == end:
                    continue
                path = []
                intermediate = end
                while intermediate != start:
                    path.append(intermediate)
                    intermediate = predecessors[intermediate]
                    if intermediate == -9999:  # Check for unreachable nodes
                        path = []
                        break
                path.reverse()
                for node in path[1:-1]:
                    betweenness[node] += 1
            
            return betweenness
        
        results = Parallel(n_jobs=-1)(delayed(compute_for_node)(i) for i in range(n))
        total_betweenness = np.sum(results, axis=0)
        total_betweenness /= 2  # to account for each path being counted twice in an undirected graph
        return total_betweenness.tolist()
    
    
    def calculate_centrality_measures(self, layer_name):
        """
        Calculate various centrality measures for a given layer, including degree centrality,
        betweenness centrality, and eigenvector centrality.
        """
        if layer_name not in self.network.layers:
            raise ValueError(f"Layer {layer_name} not found.")
        
        matrix = self.network.edges[layer_name]
        if isinstance(matrix, list):
            matrix = np.array(matrix)
        
        try :
            n = len(matrix)
        except :
            try :
                n = matrix.shape[0]
            except Exception as e :
                raise Exception(e)
        # Degree Centrality
        if sp.issparse(matrix):
            degree_centrality = matrix.sum(axis=0).A1 / (n - 1)
        else:
            degree_centrality = matrix.sum(axis=1) / (n - 1)
        
        # Betweenness Centrality
        betweenness_centrality = self._calculate_betweenness_centrality(matrix, n)
        
        # Eigenvector Centrality
        eigenvector_centrality = self._calculate_eigenvector_centrality(matrix)
        
        centralities = {
            'degree_centrality': degree_centrality.tolist(),
            'betweenness_centrality': betweenness_centrality,
            'eigenvector_centrality': eigenvector_centrality
        }
        return centralities
    
    
    def _calculate_betweenness_centrality(self, matrix, n):
        """
        Calculate betweenness centrality for each node in the network.
        """
        dist_matrix, predecessors = shortest_path(csgraph=matrix, directed=False, return_predecessors=True)
        betweenness = np.zeros(n)
        for start in range(n):
            for end in range(n):
                if start == end:
                    continue
                path = [end]
                while path[-1] != start:
                    pred = predecessors[start, path[-1]]
                    if pred == -9999:  # Unreachable
                        break
                    path.append(pred)
                path = path[::-1]  # Reverse to get the correct order
                for i in range(len(path) - 1):
                    betweenness[path[i]] += 1 / (len(path) - 1)
        betweenness /= ((n - 1) * (n - 2))
        return betweenness.tolist()
    
    
    def _calculate_eigenvector_centrality(self, matrix):
        """
        Calculate eigenvector centrality using the power iteration method.
        """
        if sp.issparse(matrix):
            eigenvalue, eigenvector = eigs(A=matrix, k=1, which='LR')
        else:
            eigenvalue, eigenvector = np.linalg.eig(matrix)
            largest = np.argmax(eigenvalue)
            eigenvector = np.real(eigenvector[:, largest])
        eigenvector_centrality = eigenvector / np.linalg.norm(eigenvector, 1)
        return eigenvector_centrality.tolist()

#end#