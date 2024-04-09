from simpleN import MultilayerNetwork, Visualize

print(" Hi, This Will generate a test or you would say example of this package ! ")

# Creating an instance of Class :
G = MultilayerNetwork()

# Adding layers :
G.add_layer('test_1')
G.add_layer('test_2')

# Adding some random nodes in range of 100 to layer one :
for node_ in range(100) :
    G.add_node(layer_name='test_1', node= node_)

# Adding some other random nodes in range of (100, 200) to layer two :
for node_ in range(100, 200) :
    G.add_node(layer_name='test_2', node= node_)

# Create some fake edge too! This is where all the node from these edges are inside one layer -layer one here- :
for i in range(0, 50, 2 ) :
    j = i + 50
    G.add_edge(node1= i, node2= j, layer_name='test_1')

# Create some fake edge too! This is where all the node from these edges are inside one layer -layer two here- :
for i in range(120, 170, 2 ) :
    j = i + 25
    G.add_edge(node1= i, node2= j, layer_name='test_2')

# Link some edges that their nodes are not in same layer !
for i in range(40, 80, 2 ) :
    for j in range( 101, 140, 2) :
        G.add_inter_layer_edge(node1= i, layer1= 'test_1', node2= j, layer2='test_2')

# Visulizing !
Visualize(G).show_graph( edge_visibility_threshold = 0.2 )