# Simple-Network Package

```bash
        pip install simple-network
```
## Overview

The Simple-Network Package is a powerful tool designed for the construction and visualization of complex, multilayer networks. With an emphasis on ease of use and flexibility, this package allows users to create intricate network structures and render them in stunning 3D using Python. The MultilayerNetwork class is designed to efficiently handle both directed and undirected graphs, support sparse representations for large graphs, and manage attributes for nodes and edges, including the capability to handle inter-layer edges.

## Core Features of the MultilayerNetwork Class

    Initialization and Graph Representation:
        The constructor initializes an empty multilayer network with optional arguments to specify whether the graph is directed and whether it is large, impacting the internal storage strategy.
        Nodes and edges are stored in dictionaries, allowing for efficient access and manipulation. Nodes are stored per layer, and edges are stored either as adjacency matrices (for small graphs) or as sparse matrices (for large graphs), depending on the graph's size.

    Layer Management:
        Layers are dynamically managed, with the ability to add layers as needed. Each layer operates independently, with its own set of nodes and edges.

    Node and Edge Management:
        Nodes can be added to specific layers, ensuring that the network's structure remains flexible and adaptable to various scenarios.
        Edges can be added within layers or between layers (inter-layer edges), supporting both intra-layer and inter-layer interactions. This flexibility is crucial for accurately modeling complex systems where entities might interact in multiple, distinct contexts.

    Attribute Management:
        The class supports setting attributes for both nodes and edges. This feature is particularly useful for annotating nodes and edges with additional information, such as weights, types, or any domain-specific data required for analysis.

    Bulk Updates:
        Methods are provided for preparing, accumulating, and applying bulk updates to edges. This functionality is designed to optimize performance when applying a large number of updates by minimizing the overhead of individual operations.

    Degree Calculation:
        The class offers a method to calculate the degrees of nodes within a layer. This calculation is critical for understanding the structure and dynamics of the network, such as identifying key nodes based on their connectivity. The method is designed to efficiently handle both dense and sparse representations and offers parallel execution for large graphs, further enhancing performance.

## Scalability and Flexibility

The design of the MultilayerNetwork class reflects a thoughtful consideration of the complexities involved in managing multilayer networks. By accommodating both dense and sparse graph representations, the class is scalable to networks of varying sizes, from small to very large graphs. Furthermore, the class's methods for managing layers, nodes, edges, and attributes provide a flexible framework that can be adapted to a wide range of applications, from social network analysis to biological network studies.


## Comprehensive Analysis class

MultilayerNetwork Analysis, is designed to perform a variety of analyses on multilayer networks. It is equipped with several methods, each tailored to analyze different aspects of a network, including its structure, connectivity, and evolution over time. 

Let's delve into the details of its functionalities and the algorithms employed:

**Functional Overview**

* Layerwise Degree Distribution:
        This function calculates the degree distribution for each layer within the network. It utilizes the degrees computed by self.network.calculate_layer_degrees(layer), normalizing them by the total number of nodes to get the distribution.

* Aggregate Network:
        Aggregates all layers into a single-layer network by summing up the weights of inter-layer edges. This unified view is useful for analyses that require a holistic perspective of the multilayer network.

* Community Detection:
        Employs spectral clustering on a specified layer to detect communities within that layer. This method is crucial for identifying groups of closely interconnected nodes, shedding light on the network's modular structure.

* Global Efficiency:
        Calculates the global efficiency for a given layer, offering insights into the network's overall integration and the efficiency of information or resource transfer across it.

* Connected Components:
        Counts the number of connected components within a specific layer. This metric helps understand the network's fragmentation or coherence.

* Dynamic Changes Analysis:
        Analyzes the evolution of global efficiency across multiple network snapshots, allowing for the observation of changes in network connectivity over time.

* Inter-Layer Connectivity Exploration:
        Investigates the patterns of connectivity between different network layers, quantifying the density of inter-layer connections and analyzing the distribution of their weights.

**Centrality Measures**

The implementation of centrality measures—degree, betweenness, and eigenvector centrality—is pivotal for understanding the roles and influences of nodes within the network:

* Degree Centrality:
        Calculated directly from the adjacency matrix, reflecting the number of connections each node has.

* Betweenness Centrality:
        Computed by identifying all shortest paths and counting how many times each node acts as a bridge along these paths. This measure highlights nodes that play crucial roles in facilitating communication within the network.

* Eigenvector Centrality:
        Relies on the principal eigenvector of the adjacency matrix to identify influential nodes, not just those with many connections, but those linked to other highly connected nodes.

**Algorithmic Details**

    The Betweenness Centrality calculation is particularly noteworthy. It utilizes the Floyd-Warshall algorithm via scipy.sparse.csgraph.shortest_path to find all shortest paths and then iterates through these paths to compute the centrality scores.

    Eigenvector Centrality employs the power iteration method or direct eigenvalue decomposition (depending on the matrix type) to find the principal eigenvector, which serves as the centrality measure.

**Practical Implications**

This class provides a robust toolkit for analyzing multilayer networks, accommodating a wide range of applications from social network analysis to studying biological networks or transportation systems. By offering insights into the structure, dynamics, and influential entities within the network, MultilayerNetworkAnalysis enables researchers and practitioners to unravel complex relational patterns, optimize connectivity, and identify key influencers or critical points within the network.


This implementation of centrality measures with Scipy provides a deeper understanding of the underlying mathematics of network analysis. It illustrates the direct manipulation of adjacency matrices and the application of fundamental linear algebra and graph theory concepts, offering a robust foundation for custom network analysis tool development.

## Prerequisites

The package is compatible with Python 3.x environments and requires the following libraries:

    NumPy
    Plotly

## Installing Dependencies

First, install the required libraries using the following command:

```bash
pip install numpy plotly
```
## Installation

To install the Simple-Network Package, ensure you have Python 3.x installed on your system. The package depends on NumPy and Plotly, which will be installed automatically if you don't have them already.

```bash
pip install Simple-Network
```
This package offers comprehensive tools for constructing and visualizing complex multilayer networks in a 3D space. It features two primary components: MultilayerNetwork, for creating the network structure, and Visualize, for rendering the network interactively in 3D.

## Usage
**Building a Multilayer Network**

To start building a multilayer network, import the MultilayerNetwork class from the package:
```python
from simpleN import MultilayerNetwork
```
*Initialize the network*
```
graph = MultilayerNetwork()
```
*Adding Layers*

```python
graph.add_layer(layer_name='Layer_1')
# Or Simple as :
graph.add_layer('Layer_1')
#Both lines above are same as each other!
```
*Adding Nodes*

```python
graph.add_node(layer_name='Layer_1', node=1)
graph.add_node(layer_name='Layer_1', node=4)
# Or Simple as :
graph.add_node(1, Layer_1')
graph.add_node(4, 'Layer_1')
#Both two ways above are same as each other!
```
*Adding Edges*
```python
graph.add_edge(node1=1, node2=4, layer_name='Layer_1', weight=1)
```

*Setting Node and Edge Attributes*
```python
graph.set_node_attribute(node=1, attr_name='attr1', attr_value='value1')

graph.set_edge_attribute(node1=1, node2=4, layer_name='Layer_1', attr_name='attr1', attr_value='value1')
```

**Visualizing the Network**

To visualize the network in 3D:

```python

from visualize import Visualize

# Create an instance of Visualize with the network
my_visualizer = Visualize(network=graph)

# Visualize the network
my_visualizer.show_graph(edge_visibility_threshold=0.1)
```
Or Simple as :

```python
Visualize(graph).show_graph()
```
## Examples

For basic usage examples, please refer to the Jupyter Notebook in the Examples folder.

## API Reference

**Class: MultilayerNetwork**

*Attributes:*

    directed (bool): Indicates whether the network edges are directed.
    node_count (int): Total number of unique nodes in the network.
    node (list): List of unique nodes.
    nodes (dict): Nodes organized by layer. Format: {layer_name: [nodes]}
    edges (dict): Edge weights organized by layer. Format: {layer_name: numpy_array}
    layers (list): List of layer names.
    node_attributes (dict): Node attributes. Format: {node: {attr_name: attr_value}}
    edge_attributes (dict): Edge attributes. Format: {(layer_name, node1, node2): {attr_name: attr_value}}
    inter_layer_edges (list): Edges between layers. Format: (node1, layer1, node2, layer2, weight)

*Methods:*

    __init__(self, directed=False): Initializes a new network. Defaults to undirected.
    add_layer(self, layer_name): Adds a new layer. If it exists, does nothing.
    add_node(self, layer_name, node): Adds a node to a layer. Creates layer if non-existent.
    add_edge(self, node1, node2, layer_name, weight=1): Adds an edge within a layer. Initializes layer if needed.
    set_node_attribute(self, node, attr_name, attr_value): Sets an attribute for a node.
    set_edge_attribute(self, node1, node2, layer_name, attr_name, attr_value): Sets an attribute for an edge.
    add_inter_layer_edge(self, node1, layer1, node2, layer2, weight=1): Adds an edge between two different layers.
    Additional methods detailed in the documentation handle internal operations, node and edge validation, and network analysis functions such as calculating node degrees within layers.

**Class: Visualize**

The Visualize class enables interactive 3D visualization of networks using Plotly.

*Methods:*

    show_graph(edge_visibility_threshold=0.1): Renders the network in 3D. Edges below the specified visibility threshold are not displayed.

## Mistakes and Corrections

To err is human, and nobody likes a perfect person! If you come across any mistakes or if you have questions, feel free to raise an issue or submit a pull request. Your contributions to improving the content are highly appreciated. Please refer to GitHub contributing guidelines for more information on how to participate in the development.

## License

This project is licensed under the MIT License. For more details, see the LICENSE file in the project repository.

## Contact

email: cloner174.org@gmail.com

other: https://t.me/PythonLearn0
