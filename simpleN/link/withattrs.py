#                                 #          In the Name of GOD   # #
#
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.preprocessing import StandardScaler


class AdvancedLinkPredictionWithAttributes:
    
    def __init__(self, multilayer_network):
        self.network = multilayer_network
        self.model = LogisticRegression(solver='liblinear')
        self.node_embeddings = None
        self.scaler = StandardScaler()
    
    
    def generate_embeddings(self, layer, n_components=None):
        """
        Generate node embeddings using Singular Value Decomposition (SVD) on the adjacency matrix of the network.
        If n_components is not specified, use the smaller dimension of the matrix minus one.
        """
        matrix = self.network.edges[layer]
        if not isinstance(matrix, csr_matrix):
            matrix = csr_matrix(matrix)
        
        # Determine the optimal number of components if not specified
        if n_components is None or n_components > min(matrix.shape):
            n_components = min(matrix.shape) - 1  # max possible components that are meaningful
        
        if n_components > 0:  # Proceed only if it's possible to decompose
            svd = TruncatedSVD(n_components=n_components)
            self.node_embeddings = svd.fit_transform(matrix)
        else:
            print("Insufficient size of the matrix for SVD.")
    
    
    def train_model(self, layer, use_cosine_similarity : bool = True):
        """
        Train a model using the embeddings combined with node attributes.
        """
        self.use_cosine_similarity = use_cosine_similarity
        nodes = list(self.network.nodes[layer])
        X = []
        y = []
        # Generate training data
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1, node2 = nodes[i], nodes[j]
                index1 = self.network.node_map[node1]['index']
                index2 = self.network.node_map[node2]['index']
                attrs1 = self.network.node_map[node1]['attributes']
                attrs1 = np.array(list(attrs1), dtype=float)
                attrs2 = self.network.node_map[node2]['attributes']
                attrs2 = np.array(list(attrs2), dtype=float)
                if use_cosine_similarity :
                    # Use indices to access embeddings
                    embedding1 = self.node_embeddings[index1].reshape(1, -1)
                    embedding2 = self.node_embeddings[index2].reshape(1, -1)
                
                    # Calculate cosine similarity
                    cos_sim = cosine_similarity(embedding1, embedding2)[0][0]
                
                    # Combine embeddings and cosine similarity into one feature vector
                    feature_vector = np.concatenate([self.node_embeddings[index1], 
                                                    self.node_embeddings[index2], 
                                                    attrs1,
                                                    attrs2,
                                                    [cos_sim]])
                else:
                    feature_vector = np.concatenate([self.node_embeddings[index1], self.node_embeddings[index2], attrs1, attrs2])
                
                X.append(feature_vector)
                # Determine if an edge exists
                if node2 in self.network.node_map[node1]['edges']['node']:
                    y.append(1)
                else:
                    y.append(0)        
        
        X = self.scaler.fit_transform(X)  # Scale features
        self.model.fit(X, y)
    
    
    def predict_link(self, node1, node2):
        """
        Predict whether a link between two nodes exists, using both embeddings and attributes.
        """
        if self.node_embeddings is None:
            raise ValueError("Node embeddings not generated. Call generate_embeddings first.")
        
        attrs1 = self.network.node_map[node1]['attributes']
        attrs2 = self.network.node_map[node2]['attributes']
        index1 = self.network.node_map[node1]['index']
        index2 = self.network.node_map[node2]['index']
        if self.use_cosine_similarity :
            embedding1 = self.node_embeddings[index1].reshape(1, -1)
            embedding2 = self.node_embeddings[index2].reshape(1, -1)
            cos_sim = cosine_similarity(embedding1, embedding2)[0][0]
            
            feature_vector = np.concatenate([self.node_embeddings[index1], 
                                            self.node_embeddings[index2], 
                                            attrs1,
                                            attrs2,
                                            [cos_sim]])
            feature_vector = self.scaler.transform([feature_vector])     
        else:
            feature_vector = np.concatenate([self.node_embeddings[index1], self.node_embeddings[index2], attrs1, attrs2])
            feature_vector = self.scaler.transform([feature_vector])  # Scale the features
        
        return self.model.predict_proba(feature_vector)[0, 1]
    
    
    def predict_links(self, layer, top_k=10):
        """
        Predict top K potential new links for a given layer using the trained model.
        """
        nodes = list(self.network.nodes[layer])
        scores = {}
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1, node2 = nodes[i], nodes[j]
                if node1 not in self.network.node_map[node2]['edges']['node'] and node2 not in self.network.node_map[node1]['edges']['node']:
                    score = self.predict_link(node1, node2)
                    scores[(node1, node2)] = score
        # Return the top k pairs with the highest scores
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

#end#