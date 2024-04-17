#                                 #          In the Name of GOD   # #
#
import plotly.graph_objects as go
import numpy as np

class Visualize:
    
    def __init__(self, network):
        self.network = network
    
    def show_graph(self,
                   space_between_layers : int = 1, 
                   edge_visibility_threshold : float = 0.1, 
                   marker_size : int = 5,
                   line_width : float = 1 , 
                   colorscale = 'Viridis', 
                   title='Network Visualization'):
        """
        The Options for colorscale is the one of these :
        ( 'aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
             'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
             'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
             'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
             'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
             'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
             'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
             'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
             'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
             'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
             'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
             'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
             'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
             'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
             'ylorrd' )
        """
        edge_x = []
        edge_y = []
        edge_z = []
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple']
        layer_positions = {layer_name: idx * space_between_layers for idx, layer_name in enumerate(self.network.layers)}
        node_positions = {}
        
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
                            edge_x.extend([x_i, x_j, None])
                            edge_y.extend([y_i, y_j, None])
                            edge_z.extend([z_i, z_j, None])
        
        for edge in self.network.inter_layer_edges:
            (node1, layer1), (node2, layer2), weight = edge
            if weight > edge_visibility_threshold:
                x1, y1, z1 = node_positions[layer1][node1]
                x2, y2, z2 = node_positions[layer2][node2]
                edge_x.extend([x1, x2, None])
                edge_y.extend([y1, y2, None])
                edge_z.extend([z1, z2, None])
        
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(size=marker_size, color=node_z, colorscale = colorscale),
            text=node_text
        )
        
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(width=line_width, color='grey'),
            hoverinfo='none'
        )
        
        layout = go.Layout(
            title=title,
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
        fig.show()

#end#