# Simple-Network Package

## Overview

The Simple-Network Package is a powerful tool designed for the construction and visualization of complex, multilayer networks. With an emphasis on ease of use and flexibility, this package allows users to create intricate network structures and render them in stunning 3D using Python. Whether for academic research, data analysis, or educational purposes, this package offers the functionality needed to bring your network visualizations to life.

## Features

- **Multilayer Network Creation**: Easily define multiple layers within your networks, adding nodes and edges with customizable attributes.
- **3D Interactive Visualization**: Utilize the integrated visualization tools to explore your networks in a dynamic 3D space.
- **Customizable Attributes**: Set and manage attributes for nodes and edges to represent complex relationships and properties.
- **Python 3.x Support**: Compatible with modern Python environments, ensuring seamless integration into your projects.


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
```
*Adding Nodes*

```python
graph.add_node(layer_name='Layer_1', node=1)
graph.add_node(layer_name='Layer_1', node=4)
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
visualizer = Visualize(network=graph)

# Visualize the network
visualizer.show_graph(edge_visibility_threshold=0.1)
```
Alternatively:

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

## Contributing

We welcome contributions to the MultilayerNetwork Package. Please refer to GitHub contributing guidelines for more information on how to participate in the development.

## License

This project is licensed under the MIT License. For more details, see the LICENSE file in the project repository.

## Contact

email: cloner174.org@gmail.com

other: https://t.me/PythonLearn0
