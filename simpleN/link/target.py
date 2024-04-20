#                                 #          In the Name of GOD   # #
#
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import warnings


class NodeAttributePredictor:
    
    def __init__(self, network, use_embeddings : bool = False ):
        
        self.network = network
        self.model = RandomForestRegressor(n_estimators=100)
        self.scaler = StandardScaler()
        self.use_embeddings = use_embeddings
        if self.use_embeddings:
            self.embeddings = None
            warnings.warn( " You should Run the generate_embeddings method Now \n The Options for method in that method are : concatenate or average or pca. Default is average")
    
    
    def generate_embeddings(self, layers : str | list = None, method : str = 'average' , n_components : int = None ):
        """
        Generate combined embeddings for nodes across multiple layers.
        """
        from .advance import AdvanceLinkPrediction
        layer_embeddings = []
        layers_ = self.network.layers if layers is None else layers
        if isinstance(layers_ , list ) :
            for layer in layers_:
                # Generate embeddings for each layer
                # AdvanceLinkPrediction -> ALP
                ALP = AdvanceLinkPrediction(self.network)
                ALP.generate_embeddings(layer=layer, n_components = n_components)
                if ALP.node_embeddings is not None:
                    layer_embeddings.append( ALP.node_embeddings )
                else:
                    pass
            if method == 'concatenate':
                self.embeddings = np.concatenate(layer_embeddings, axis=1)
            elif method == 'average':
                self.embeddings = np.mean(np.stack(layer_embeddings, axis=-1), axis=-1)
            elif method == 'pca':
                concatenated = np.concatenate(layer_embeddings, axis=1)
                pca = PCA(n_components=min(concatenated.shape[1], 100))  # Reduce to 100 components or fewer
                self.embeddings = pca.fit_transform(concatenated)
        else:
            ALP = AdvanceLinkPrediction(self.network)
            ALP.generate_embeddings(layer=layers_, n_components = n_components)
            self.embeddings = ALP.node_embeddings
    
    
    def prepare_data(self):
        X = []
        y = []
        for node, data in self.network.node_map.items():
            if 'attributes' in data and 'additional' in data:
                attributes = np.array(list(data['attributes']), dtype=float)
                if self.use_embeddings:
                    embedding = self.embeddings[data['index']]
                    features = np.concatenate([attributes, embedding])
                else:
                    features = attributes
                X.append(features)
                y.append(data['additional'])
        return np.array(X), np.array(y)
    
    
    def perform_grid_search(self, 
                            layer : str = None,           # Not Implantted Yet!
                            n_estimators : list = [50, 100, 200],
                            max_features : list = ['auto', 'sqrt', 'log2'],
                            max_depth : list = [None, 10, 20, 30],
                            min_samples_split : list = [2, 5, 10],
                            min_samples_leaf : list = [1, 2, 4],
                            cv = 5 ,
                            scoring = 'neg_mean_squared_error',
                            verbose : int = 1):
        
        # Parameters for Grid Search
        param_grid = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
        }
        
        X, y = self.prepare_data()
        X = self.scaler.fit_transform(X)  # Normalize features
        
        model = RandomForestRegressor()
        
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring = scoring, verbose = verbose)
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_  # Update model to the best found
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best score achieved: {grid_search.best_score_}")
    
    
    def perform_random_search(self,
                            layer : str = None,           # Not Implantted Yet!
                            n_estimators : list = [50, 100, 200],
                            max_features : list = ['auto', 'sqrt', 'log2'],
                            max_depth : list = [None, 10, 20, 30],
                            min_samples_split : list = [2, 5, 10],
                            min_samples_leaf : list = [1, 2, 4],
                            n_iter = 100, 
                            cv = 5 ) :
        # Parameters for Random Search
        param_search = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
        }        
        X, y = self.prepare_data()
        X = self.scaler.fit_transform(X)
        model = RandomForestRegressor()
        
        random_search = RandomizedSearchCV(model, param_search, n_iter=n_iter, cv=cv, scoring='neg_mean_squared_error', verbose=1, random_state=42)
        random_search.fit(X, y)
        self.model = random_search.best_estimator_
        print(f"Best parameters found: {random_search.best_params_}")
        print(f"Best score achieved: {random_search.best_score_}")
    
    
    def train_and_evaluate(self, test_size=0.2, random_state=42):
        X, y = self.prepare_data()
        X = self.scaler.fit_transform(X)  # Normalize features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Train the model
        self.model.fit(X_train, y_train)
        print("Model trained successfully!")
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R^2 Score: {r2:.2f}")
        return self.model
    
    
    def predict_additional(self, attributes : list , embedding=None):
        if self.use_embeddings and embedding is not None:
            attributes = np.concatenate([np.array(attributes, dtype=float), embedding])
        else:
            attributes = np.array(attributes, dtype=float)
        
        attributes = self.scaler.transform([attributes])  # Normalize features
        return self.model.predict(attributes)[0]

#end#
