#                                 #          In the Name of GOD   # #
#
import numpy as np
from concurrent.futures import ProcessPoolExecutor

class MultilayerNetwork:
    
    def __init__(self, directed=False):
        self.directed = directed
        self.node_count = 0
        self.node = [] # Retained as a list for flexibility
        self.nodes = {} # Format: {layer_name: [ A List Contianed Nodes of this layer ]}
        self.edges = {} # Format: {layer_name: numpy_array for adjacency matrix}
        self.layers = [] # Retained as a list for flexibility
        self.node_attributes = {}  # Format: {node: {attr_name: attr_value}}
        self.edge_attributes = {}  # Format: { (layer_name, node1, node2) : {attr_name : attr_value} }
        self.inter_layer_edges = []  # Retained as a list for flexibility
    
    def initial(self,
                Do_node_count : bool = False , 
                node_count_for_layer : bool = False,
                layer_name_for_counting = None,
                node = None,
                edge : tuple = None , 
                layer = None ) :
        if Do_node_count == True :
            if node_count_for_layer == False :
                for any_node in self.nodes.values() :
                    self.node_count += len(any_node)
                return
            else: 
                if layer_name_for_counting is None :
                    raise ValueError( "You should pass a layer to do counting for layer mode ! obvisly! ")
                else:
                    return len(self.nodes[layer_name_for_counting])
        else:
            pass
        if node is not None :
            if len(self.node) > 0 :
                if node not in self.node:
                    raise ValueError("Node does not exist.")
                else:
                    return
        if edge is not None :
            node_1 = edge[0]
            node_2 = edge[1]
            if len(self.node) > 0 :
                if node_1 not in self.node:
                    raise ValueError("Node does not exist.")
                elif node_2 not in self.node:
                    raise ValueError("Node does not exist.")
                else:
                    return                
        if layer is not None :
            if layer not in self.layers :
                raise ValueError(f" {layer} layer does not exist.")
            else:
                return
    
    def add_layer(self, layer_name):#np.zeros((self.node_count, self.node_count))
        if layer_name not in self.layers:
            self.layers.append(layer_name)
            self.nodes[layer_name] = []
        else:
            print( f" Layer with this name {layer_name} already Exist ! ")
    
    def add_node(self, layer_name, node ) :
        if layer_name not in self.layers :
            self.add_layer(layer_name)
        else:
            pass
        if node not in self.nodes[layer_name] :
            self.nodes[layer_name].append(node)
            self.node.append(node)
        else:
            print( f" Layer with this name {layer_name} already have this node {node} ! ")
    
    def set_node_attribute(self, node, attr_name, attr_value):
        try:
            self.initial(node= node)
        except Exception as e:
            raise Exception(e)
        if node not in self.node_attributes.keys():
            self.node_attributes[node] = {}
        else:
            print(" The attr for this node already saved! if you are want to change it, You should start again with class")
        self.node_attributes[node][attr_name] = attr_value
    
    def set_edge_attribute(self, node1, node2, layer_name, attr_name, attr_value):
        try:
            self.initial(edge=(node1, node2))
            self.initial(layer = layer_name)
        except Exception as e:
            raise Exception(e)
        if self.directed:
            edge_key = (layer_name, node1, node2)
        else:
            edge_key = (layer_name, min(node1, node2), max(node1, node2))
        if edge_key not in self.edge_attributes:
            self.edge_attributes[edge_key] = {}
        self.edge_attributes[edge_key][attr_name] = attr_value
    
    def add_edge(self, node1, node2, layer_name, weight=1):
        try:
            self.initial(edge=(node1, node2))
            self.initial(layer = layer_name)
        except Exception as e:
            raise Exception(e)
        self.initial(Do_node_count= True)
        if layer_name not in self.edges.keys() :
            self.edges[layer_name] = np.zeros((self.node_count, self.node_count))
        else:
            self.edges[layer_name][node1, node2] = weight
        if not self.directed:
            self.edges[layer_name][node2, node1] = weight
    
    def add_inter_layer_edge(self, node1, layer1, node2, layer2, weight=1):
        try:
            self.initial(edge= (node1, node2))
            self.initial(layer=layer1)
            self.initial(layer=layer2)
        except Exception as e:
            raise Exception(e)
        self.inter_layer_edges.append((node1, layer1, node2, layer2, weight))
    
    def calculate_layer_degrees(self, layer_name):
        if layer_name not in self.layers:
            raise ValueError(f"Layer {layer_name} not found.")
        layer_matrix = self.layers[layer_name]
        self.initial(Do_node_count = True)
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self._calculate_degree_of_node, [(layer_matrix, i) for i in range(self.node_count)]))
        return results
    
    @staticmethod
    def _calculate_degree_of_node(args):
        layer_matrix, node_index = args
        return np.sum(layer_matrix[node_index] > 0)

#end#