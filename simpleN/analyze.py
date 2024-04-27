#                                 #          In the Name of GOD   # #
#
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from joblib import Parallel, delayed
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import connected_components, shortest_path, dijkstra




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
            degrees = self.network.calculate_layer_degrees(layer).astype(int)
            degree_distributions[layer] = np.bincount(degrees) / float(len(degrees))
        
        return degree_distributions
    
    
    def aggregate_network(self, 
                          neet : bool = True, 
                          return_aggregated_network : bool = True, 
                          return_adjacency_matric : bool = False ):
        """
        Aggregate the multilayer network into a single-layer network.
        This method combines all layers into one, summing up the weights of inter-layer edges.
        """
        from .net import MultilayerNetwork
        if neet:
            aggregated_network = MultilayerNetwork()
            aggregated_network.add_layer()
            for node_ in self.network.node_set :
                aggregated_network.add_node(node = node_)
        
            for edges_ in self.network.extra_edges:
                aggregated_network.add_edge( node1= edges_[0], node2= edges_[1], weight=1 )
        
            for node_, attrs_ in self.network.node_attributes.items() :
                for attrs_name, attrs_values in attrs_.items() :
                    aggregated_network.set_node_attribute( node= node_, attr_name = attrs_name , attr_value = attrs_values )
        
            if return_aggregated_network == True and return_adjacency_matric == False :
                return aggregated_network
            elif return_aggregated_network == False and return_adjacency_matric == True :
                aggregated_matrix = aggregated_network.edges            
                return aggregated_matrix['ALL']
            elif return_adjacency_matric == True and return_aggregated_network == True :
                aggregated_matrix = aggregated_network.edges            
                return aggregated_network, aggregated_matrix['ALL']
            else :
                print( " No Action To Could Provide Output !")
                return
        else:
            aggregated_network = MultilayerNetwork()
            sources = []
            targets = []
            for edges_ in self.network.extra_edges:
                sources.append(edges_[0])
                targets.append(edges_[1])
            for node_ in range( len(self.network.node_set) ) :
                aggregated_network.add_node(node = node_)
            
            if len(sources) != len(targets) :
                raise BlockingIOError(" Opps ! No way ! ")
            for i in range(len(sources)) :
                source_node = sources[i]
                target_node = targets[i]
                
                aggregated_network.add_edge( 
                                            node1 = self.network.node_map[source_node]['index'], 
                                            node2 = self.network.node_map[target_node]['index'], 
                                            weight = 1 )
            for node_, attrs_ in self.network.node_attributes.items() :
                for attrs_name, attrs_values in attrs_.items() :
                    aggregated_network.set_node_attribute( 
                                                          node = self.network.node_map[node_]['index'] , 
                                                          attr_name = attrs_name , 
                                                          attr_value = attrs_values )
            del sources
            del targets
            if return_aggregated_network == True and return_adjacency_matric == False :
                return aggregated_network
            elif return_aggregated_network == False and return_adjacency_matric == True :
                aggregated_matrix = aggregated_network.edges            
                return aggregated_matrix['ALL']
            elif return_adjacency_matric == True and return_aggregated_network == True :
                aggregated_matrix = aggregated_network.edges            
                return aggregated_network, aggregated_matrix['ALL']
            else :
                print( " No Action To Could Provide Output !")
                return
    
    def detect_communities(self, layer_name, n_clusters=2):
        """
        Detect communities within a specific layer using spectral clustering.
        """
        if layer_name not in self.network.layers:
            raise ValueError(f"Layer {layer_name} not found. Ensure the layer name is correct.")
        
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
        
        weights = [weight for _, _, weight in inter_layer_edges]
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
    
    
    def calculate_centrality_measures(self, layer_name, use_weight=False):
        
        if layer_name not in self.network.layers:
            raise ValueError(f"Layer {layer_name} not found.")
        
        matrix = self.network.edges[layer_name]
        
        if isinstance(matrix, list):
            matrix = np.array(matrix)
        if not sp.issparse(matrix):
            matrix = sp.csr_matrix(matrix)
        
        # Degree Centrality
        if sp.issparse(matrix):
            if use_weight:
                degree_centrality = matrix.sum(axis=0).A1 / (matrix.shape[0] - 1)
            else:
                degree_centrality = (matrix != 0).sum(axis=0).A1 / (matrix.shape[0] - 1)
        else:
            if use_weight:
                degree_centrality = matrix.sum(axis=1) / (matrix.shape[0] - 1)
            else:
                degree_centrality = (matrix != 0).sum(axis=1) / (matrix.shape[0] - 1)
        
        # Betweenness Centrality
        betweenness_centrality = self._calculate_betweenness_centrality(matrix, matrix.shape[0], use_weight)
        
        # Eigenvector Centrality
        eigenvector_centrality = self._calculate_eigenvector_centrality(matrix)
        
        centralities = {
            'degree_centrality': degree_centrality.tolist(),
            'betweenness_centrality': betweenness_centrality,
            'eigenvector_centrality': eigenvector_centrality
        }
        return centralities
    
    
    def calculate_centrality_with_attributes(self, layer_name, attribute_name, use_weight=False):
        
        if layer_name not in self.network.layers:
            raise ValueError(f"Layer {layer_name} not found.")
        
        matrix = self.network.edges[layer_name]
        if isinstance(matrix, list):
            matrix = np.array(matrix)
        if not sp.issparse(matrix):
            matrix = sp.csr_matrix(matrix)
        
        # Retrieve node attributes
        attributes = [self.network.node_attributes.get(node, {}).get(attribute_name, 1) for node in self.network.nodes[layer_name]]
        
        # Adjust matrix for attributes if using weights
        if use_weight:
            attr_matrix = sp.diags(attributes)
            matrix = attr_matrix @ matrix if sp.issparse(matrix) else np.diag(attributes) @ matrix
        
        # Degree Centrality with attributes
        degree_centrality = matrix.sum(axis=0).A1 / (matrix.shape[0] - 1) if sp.issparse(matrix) else matrix.sum(axis=1) / (matrix.shape[0] - 1)
        
        # Betweenness Centrality with attributes
        betweenness_centrality = self._calculate_betweenness_centrality(matrix, matrix.shape[0], use_weight)
        
        # Eigenvector Centrality with attributes
        eigenvector_centrality = self._calculate_eigenvector_centrality(matrix)
        
        centralities = {
            'degree_centrality': degree_centrality.tolist(),
            'betweenness_centrality': betweenness_centrality,
            'eigenvector_centrality': eigenvector_centrality
        }
        
        return centralities
    
    
    def _calculate_betweenness_centrality(self, matrix, n, use_weight=False):
        """
        Calculate the betweenness centrality for each node in a weighted or unweighted graph.
        """
        if use_weight:
            # Use Dijkstra's algorithm for weighted graphs
            dist_matrix, predecessors = shortest_path(csgraph=matrix, directed=self.network.directed, return_predecessors=True, unweighted=False)
        else:
            # Unweighted graph, use faster Floyd-Warshall algorithm
            dist_matrix, predecessors = shortest_path(csgraph=matrix, directed=self.network.directed, return_predecessors=True, unweighted=True)
        
        betweenness = np.zeros(n)
        for source in range(n):
            for target in range(n):
                if source != target:
                    # Reconstruct the shortest path from source to target
                    path = []
                    intermediate = target
                    while predecessors[source, intermediate] != -9999:
                        path.append(intermediate)
                        intermediate = predecessors[source, intermediate]
                        if intermediate == source:
                            break
                    path.append(source)
                    path.reverse()
                    
                    # Count the betweenness
                    for v in path[1:-1]:  # exclude the source and target themselves
                        betweenness[v] += 1
        
        # Normalize the betweenness scores
        if not self.network.directed:
            betweenness /= 2
        return betweenness.tolist()
    
    
    def _calculate_eigenvector_centrality(self, matrix, use_weight=False):
        """
        Calculate eigenvector centrality using the power iteration method.
        Handles both sparse and dense matrix formats. Provides an option to consider or ignore edge weights.
        
        Args:
        matrix (np.ndarray or sp.spmatrix): The adjacency matrix of the network.
        use_weight (bool): If True, use the edge weights as given; if False, treat the graph as unweighted.
        """
        try:
            if not use_weight:
                # Convert all non-zero entries to 1 to ignore actual weights
                if sp.issparse(matrix):
                    matrix = sp.csr_matrix((np.ones_like(matrix.data), matrix.indices, matrix.indptr), shape=matrix.shape)
                else:
                    matrix = np.where(matrix != 0, 1, 0)
            
            if sp.issparse(matrix):
                matrix = matrix.astype(np.float64)  # Ensure matrix is of floating-point type
                eigenvalue, eigenvector = eigs(A=matrix, k=1, which='LR', maxiter=10000, tol=1e-6)
            else:
                matrix = np.array(matrix, dtype=np.float64)  # Ensure matrix is of floating-point type
                eigenvalue, eigenvector = np.linalg.eig(matrix)
                largest = np.argmax(eigenvalue)
                eigenvector = eigenvector[:, largest]
            
            eigenvector_centrality = np.abs(np.real(eigenvector)) / np.linalg.norm(np.real(eigenvector), 1)
            return eigenvector_centrality.tolist()
        
        except Exception as e:
            error_message = f"Failed to calculate eigenvector centrality. Ensure the matrix is appropriate for this operation. Error: {e}"
            raise RuntimeError(error_message)

#end#
