#                                 #          In the Name of GOD   # #
#
import numpy as np
import scipy.sparse as sp
from concurrent.futures import ProcessPoolExecutor

class MultilayerNetwork:
    
    def __init__(self, directed : bool = False, large_graph : bool = False):
        
        self.directed = directed
        self.large_graph = large_graph
        self.node_set = set()
        self.nodes = {}#                   Nodes by layer
        self.edges = {}#                   Adjacency matrices or arrays by layer
        self.layers = []#                  List of layer names
        self.node_attributes = {}
        self.edge_attributes = {}
        self.inter_layer_edges = []#       Managing inter-layer edges
        self.extra_edges = []
        self.extra_attributes = []
    
    
    def add_layer(self, layer_name : str ):
        
        if layer_name not in self.layers:
            
            self.layers.append(layer_name)
            self.nodes[layer_name] = []
            self.edges[layer_name] = []
    
    
    def add_node(self, layer_name : str , node ):
        
        if layer_name not in self.layers:
            self.add_layer(layer_name)
        
        if node not in self.nodes[layer_name]:
            self.nodes[layer_name].append(node)
            self.node_set.add(node)
            
            if self.large_graph:
                if layer_name not in self.edges or not isinstance(self.edges[layer_name], sp.lil_matrix):
                    self.edges[layer_name] = sp.lil_matrix((1, 1))
            else:
                if layer_name not in self.edges or not isinstance(self.edges[layer_name], np.ndarray):
                    self.edges[layer_name] = np.zeros((1, 1), dtype=int)
                
                old_matrix = self.edges[layer_name]
                new_size = len(self.nodes[layer_name])
                new_matrix = np.zeros((new_size, new_size), dtype=int)
                new_matrix[:old_matrix.shape[0], :old_matrix.shape[1]] = old_matrix
                self.edges[layer_name] = new_matrix
    
    
    def add_edge(self, node1, node2, layer_name1 : str, layer_name2 : str = None, weight : int | float = 1 ):
        
        if layer_name2 is None or layer_name1 == layer_name2:
            
            if layer_name1 not in self.layers:
                raise ValueError(f"Layer '{layer_name1}' does not exist.")
            
            if node1 not in self.nodes[layer_name1] or node2 not in self.nodes[layer_name1]:
                raise ValueError("One or both nodes do not exist in the specified layer.")
            
            
            tuple_like = (node1, node2)
            self.extra_edges.append( tuple_like )
            self._ensure_correct_matrix_size(layer_name1, node1, node2)
            node1_index = self.nodes[layer_name1].index(node1)
            node2_index = self.nodes[layer_name1].index(node2)
            
            self.edges[layer_name1][node1_index, node2_index] = weight
            
            if not self.directed:
                self.edges[layer_name1][node2_index, node1_index] = weight
        
        else:
            self.add_inter_layer_edge(node1, node2, layer_name1, layer_name2, weight)
    
    
    def add_inter_layer_edge(self, node1, node2, layer_name1 : str , layer_name2 : str , weight : int | float = 1 ):
        
        if layer_name1 not in self.layers or layer_name2 not in self.layers:
            raise ValueError(f"One or both specified layers ('{layer_name1}', '{layer_name2}') do not exist.")
        
        if node1 not in self.nodes.get(layer_name1, []) or node2 not in self.nodes.get(layer_name2, []):
            raise ValueError("One or both nodes do not exist in their specified layers.")
        
        self.inter_layer_edges.append(((node1, layer_name1), (node2, layer_name2), weight))
    
    
    def _ensure_correct_matrix_size(self, layer_name : str , node1, node2):
        
        node1_index = self.nodes[layer_name].index(node1)
        node2_index = self.nodes[layer_name].index(node2)
        
        if self.large_graph:
            self._ensure_matrix_initialized_and_resized(layer_name, node1_index, node2_index)
        else:
            self._ensure_dense_matrix_resized(layer_name, node1_index, node2_index)
    
    
    def _ensure_matrix_initialized_and_resized(self, layer_name : str , node1_index, node2_index):
        
        if layer_name not in self.edges or self.edges[layer_name].shape == (0, 0):
            self.edges[layer_name] = sp.lil_matrix((1, 1))
        
        current_matrix = self.edges[layer_name]
        max_index = max(node1_index, node2_index) + 1
        
        if max_index > current_matrix.shape[0]:
            new_matrix = sp.lil_matrix((max_index, max_index))
            new_matrix[:current_matrix.shape[0], :current_matrix.shape[1]] = current_matrix
            self.edges[layer_name] = new_matrix
    
    
    def _ensure_dense_matrix_resized(self, layer_name : str, node1_index, node2_index):
        
        if layer_name not in self.edges or not len(self.edges[layer_name]):
            self.edges[layer_name] = np.zeros((1, 1), dtype=int)
        
        current_matrix = np.array(self.edges[layer_name])
        max_index = max(node1_index, node2_index) + 1
        
        if max_index > current_matrix.shape[0]:
            new_matrix = np.zeros((max_index, max_index), dtype=int)
            new_matrix[:current_matrix.shape[0], :current_matrix.shape[1]] = current_matrix
            
            self.edges[layer_name] = new_matrix.tolist() if isinstance(self.edges[layer_name], list) else new_matrix
    
    
    def get_inter_layer_edges(self):
        
        return self.inter_layer_edges
    
    
    def find_inter_layer_edges(self, node, layer : str = None):
        """
        Find all inter-layer edges connected to a given node, optionally within a specific layer.
        """
        if layer:
            return [(n1, l1, n2, l2, w) for (n1, l1, n2, l2, w) in self.inter_layer_edges if (node == n1 and layer == l1) or (node == n2 and layer == l2)]
        else:
            return [(n1, l1, n2, l2, w) for (n1, l1, n2, l2, w) in self.inter_layer_edges if node == n1 or node == n2]
    
    
    def prepare_for_bulk_update(self, layer_name : str ):
        """
        Prepare or reset the accumulator for a layer.
        """
        self.bulk_updates[layer_name] = {'rows': [], 'cols': [], 'data': []}
    
    
    def accumulate_edge_update(self, node1, node2, layer_name : str, weight=1):
        """
        Accumulate an edge update for later bulk application.
        """
        if layer_name not in self.nodes:
            raise ValueError("Layer does not exist.")
        if node1 not in self.nodes[layer_name] or node2 not in self.nodes[layer_name]:
            raise ValueError("One or both nodes do not exist in the specified layer.")
        
        node1_index = self.nodes[layer_name].index(node1)
        node2_index = self.nodes[layer_name].index(node2)
        
        self.bulk_updates[layer_name]['rows'].append(node1_index)
        self.bulk_updates[layer_name]['cols'].append(node2_index)
        self.bulk_updates[layer_name]['data'].append(weight)
    
    
    def update_edge_weight(self, node1, node2, layer_name : str, new_weight : int | float ):
        
        if layer_name not in self.layers or node1 not in self.nodes[layer_name] or node2 not in self.nodes[layer_name]:
            raise ValueError("Layer, node1, or node2 does not exist.")
        
        node1_index = self.nodes[layer_name].index(node1)
        node2_index = self.nodes[layer_name].index(node2)
        self.edges[layer_name][node1_index, node2_index] = new_weight
        
        if not self.directed:
            self.edges[layer_name][node2_index, node1_index] = new_weight
    
    
    def apply_bulk_updates(self, layer_name : str ):
        """ 
        Apply accumulated updates in bulk to the adjacency matrix of a layer. 
        """
        if layer_name not in self.bulk_updates:
            return  # No updates to apply
        
        update_data = self.bulk_updates[layer_name]
        if self.large_graph:
            # for large (sparse) graphs
            update_matrix = sp.coo_matrix((update_data['data'], (update_data['rows'], update_data['cols'])),
                                        shape=self.edges[layer_name].shape)
            self.edges[layer_name] += update_matrix.tocsr()
        else:
        # for small (dense) graphs
            for row, col, data in zip(update_data['rows'], update_data['cols'], update_data['data']):
                self.edges[layer_name][row, col] = data
                if not self.directed:
                    self.edges[layer_name][col, row] = data
        
        # reset the accumulator for the layer
        self.prepare_for_bulk_update(layer_name)
    
    
    def set_node_attribute(self, node, attr_name : str , attr_value : int | float | list ):
        """
        Set or update an attribute for a node.
        Args:
        node (str or int): The node identifier.
        attr_name (str): The name of the attribute.
        attr_value (any): The value of the attribute.
        """
        if node not in self.node_set:
            raise ValueError("Node does not exist.")
        if node not in self.node_attributes:
            self.node_attributes[node] = {}
        
        self.node_attributes[node][attr_name] = attr_value
    
    
    def set_edge_attribute(self, node1, node2, layer_name : str, attr_name : str , attr_value : int | float ):
        
        if node1 not in self.node_set or node2 not in self.node_set:
            raise ValueError("One or both nodes do not exist.")
        if layer_name not in self.layers:
            raise ValueError(f"Layer {layer_name} does not exist.")
        
        edge_key = (node1, node2, layer_name) if self.directed else (min(node1, node2), max(node1, node2), layer_name)
        
        self.edge_attributes[edge_key] = self.edge_attributes.get(edge_key, {})
        self.edge_attributes[edge_key][attr_name] = attr_value
    
    
    def calculate_layer_degrees(self, layer_name : str , parallel_threshold=10000):
        
        if layer_name not in self.layers:
            raise ValueError(f"Layer {layer_name} not found.")
        layer_matrix = self.edges[layer_name]
        
        node_count = layer_matrix.shape[0] if self.large_graph else len(layer_matrix)
        
        if self.large_graph and node_count > parallel_threshold:
            with ProcessPoolExecutor() as executor:
                func_args = [(self.large_graph, layer_matrix, i) for i in range(node_count)]
                degrees = list(executor.map(MultilayerNetwork._calculate_degree_of_node, func_args))
        else:
            degrees = self._calculate_layer_degrees_single_threaded(layer_name)
        
        return degrees
    
    def get_node_attributes(self) :
        
        sorted_nodes = list( self.node_attributes.keys() )
        sorted_nodes.sort()
        
        self_node_attributes = {}    # This is just a copy (local one) of actual self.node_attributes But It is Sorted !
        for node_ in sorted_nodes :
            for node_name, attrs_dict in self.node_attributes.items() :
                
                if node_name == node_ :
                    self_node_attributes[node_name] = attrs_dict
                else:
                    continue
        
        return self_node_attributes
    
    
    def _calculate_layer_degrees_single_threaded(self, layer_name : str ):
        
        layer_matrix = self.edges[layer_name]
        
        if self.large_graph:
            
            if self.directed:
                in_degrees = layer_matrix.sum(axis=0).A1  # Column sum for in-degree
                out_degrees = layer_matrix.sum(axis=1).A1  # Row sum for out-degree
                return in_degrees, out_degrees
            
            else:
                degrees = layer_matrix.sum(axis=1).A1  # Row sum suffices
                return degrees
        else:
            degrees = np.sum(layer_matrix > 0, axis=1)  # For dense matrices
            return degrees
    
    
    @staticmethod
    def _calculate_degree_of_node(args):
        
        large_graph, layer_matrix, node_index = args
        
        if large_graph:
            # In this case layer_matrix is a sparse matrix
            row = layer_matrix.getrow(node_index)
            return row.getnnz()
        
        else:
            # In this case layer_matrix is a dense numpy array
            return np.sum(layer_matrix[node_index] > 0)

#end#