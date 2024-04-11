#                                 #          In the Name of GOD   # #
#
import plotly.graph_objects as go
import numpy as np

class Visualize:
    
    def __init__(self, network):
        self.network = network
    
    def show_graph(self, edge_visibility_threshold=0.1):
        edge_x = []
        edge_y = []
        edge_z = []
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple']
        layer_positions = {layer_name: idx * 2 for idx, layer_name in enumerate(self.network.layers)}
        node_positions = {}  # This will store the (x, y) positions for each node by layer
        # Plot nodes and add edges
        for layer_idx, (layer_name, nodes) in enumerate(self.network.nodes.items()):
            z_pos = layer_positions[layer_name]
            x_positions = np.random.rand(len(nodes))
            y_positions = np.random.rand(len(nodes))
            node_x.extend(x_positions)
            node_y.extend(y_positions)
            node_z.extend([z_pos] * len(nodes))
            node_text.extend([f'{node} ({layer_name})' for node in nodes])
            node_positions[layer_name] = {node: (x, y, z_pos) for node, x, y in zip(nodes, x_positions, y_positions)}
            if layer_name in self.network.edges:
                for i, node_i in enumerate(nodes):
                    for j, node_j in enumerate(nodes):
                        if i != j and self.network.edges[layer_name][i, j] > edge_visibility_threshold:
                            x_i, y_i, z_i = node_positions[layer_name][node_i]
                            x_j, y_j, z_j = node_positions[layer_name][node_j]
                            edge_x.extend([x_i, x_j, None])  # None will create a gap between the edges, necessary for Plotly
                            edge_y.extend([y_i, y_j, None])
                            edge_z.extend([z_i, z_j, None])
        # Inter-layer edges
        for edge in self.network.inter_layer_edges:
            (node1, layer1), (node2, layer2), weight = edge
            if weight < edge_visibility_threshold:
                continue
            x1, y1, z1 = node_positions[layer1][node1]
            x2, y2, z2 = node_positions[layer2][node2]
            edge_x.extend([x1, x2, None])
            edge_y.extend([y1, y2, None])
            edge_z.extend([z1, z2, None])
        # Create a scatter plot for nodes
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z, 
            mode='markers', 
            marker=dict(size=5, color=node_z, colorscale='Viridis'), 
            text=node_text
        )
        # Create a scatter plot for edges
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z, 
            mode='lines', 
            line=dict(width=1, color='grey'), 
            hoverinfo='none'
        )
        # Create the layout for the plot
        layout = go.Layout(
            title='Network Visualization',
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
        # Render the figure
        fig.show()

#end#