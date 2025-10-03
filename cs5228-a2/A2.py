import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


class MyRandomForestRegressor:
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.estimators = []
        
        
    def bootstrap_sampling(self, X, y):
        X_bootstrap, y_bootstrap = None, None

        N, d = X.shape

        #########################################################################################
        ### Your code starts here ###############################################################

        

        ### Your code ends here #################################################################
        #########################################################################################

        return X_bootstrap, y_bootstrap
    
        
    def feature_sampling(self, X):
        N, d = X.shape

        X_feature_sampled, indices_sampled = None, None

        #########################################################################################
        ### Your code starts here ###############################################################
        

        
        ### Your code ends here #################################################################
        #########################################################################################    

        return X_features_sampled, indices_sampled
    
    
    def fit(self, X, y):
        
        self.estimators = []
        
        for _ in range(self.n_estimators):
            
            regressor, indices_sampled = None, None
            
            #########################################################################################
            ### Your code starts here ###############################################################

            
    
            ### Your code ends here #################################################################
            #########################################################################################    
        
            self.estimators.append((regressor, indices_sampled))
            
        return self
            
            
    def predict(self, X):
        
        predictions = []
        
        #########################################################################################
        ### Your code starts here ###############################################################

        
        
        ### Your code ends here #################################################################
        #########################################################################################        
        
        return predictions
    




class AdaBoostTreeClassifier:
    
    def __init__(self, n_estimators=50):
        self.estimators, self.alphas = [], []
        self.n_estimators = n_estimators
        self.classes = None
        
    
    def fit(self, X, y):
        """
        Trains the AdaBoost classifier using Decision Trees as Weak Learners.

        Inputs:
        - X: A numpy array of shape (N, D) containing N data samples presented by D features,
        - y: A numpy array of shape (N,) containing N numerical values representing class labels
             
        Returns:
        - self
        """        
        
        N = X.shape[0]
        # Initialize the first sample as the input
        D, d = X, y
        # Initialize the sample weights uniformly
        w = np.full((N,), 1/N)
        # Create the list of class labels from all unique values in y
        self.classes = np.unique(y)
        
        
        sample_idx = np.arange(N)
        
        for m in range(self.n_estimators):

            # Step 1: Train Weak Learner on current datset sample
            estimator = DecisionTreeClassifier(max_depth=1).fit(D, d)
            
            # Add current stump to sequences of all Weak Learners
            self.estimators.append(estimator)
            
            ################################################################################
            ### Your code starts here ######################################################
            
            # Step 2: Identify all samples in X that get misclassified with the current estimator

        
            # Step 3: Calculate the total error for current estimator

        
            # Step 4: Calculate amount of say for current estimator and keep track of it
            # (we need the amounts of say later for the predictions)

        
            # Step 5: Update the sample weights w based on amount of say a

        
            # Step 6: Sample next D and d based on new weights w

        
            ### Your code ends here ########################################################
            ################################################################################            
            
        # Convert the amounts-of-say to a numpy array for convenience
        # We need this later for making our predictions
        self.alphas = np.array(self.alphas)
        
        ## Return AdaBoostTreeClassifier object
        return self         
        

            
    def predict(self, X):
        """
        Predicts the class label for an array of data points

        Inputs:
        - X: A numpy array of shape (N, D) containing N data samples presented by D features,
             
        Returns:
        - y_pred: A numpy array of shape (N,) containing N integer values representing the predicted class labels
        """        
        
        # Predict the class labels for each sample individually
        return np.array([ self.predict_sample(x) for x in X ])
        
        
    def predict_sample(self, x):
        """
        Predicts the class label for a single data point

        Inputs:
        - x: A numpy array of shape (D, ) containing D features,
             
        Returns:
        - y_pred: integer value representing the predicted class label
        """        
        
        y = None
        
        # The predict method of our classifier expects a matrix,
        # so we need to convert our sample to a NumPy array (in case it already isn't one)
        x = np.array([x])
        
        # Create a vector for all data points and all n_estimator predictions
        y_estimators = np.full(self.n_estimators, -1, dtype=np.int16)
        
        # Stores the score for each class label, e.g.,
        # class_scores[0] = class score for class 0
        # class_scores[1] = class score for class 1
        # ...
        class_scores = np.zeros(len(self.classes))

        y_pred = None
        ################################################################################
        ### Your code starts here ######################################################        
        


        ### Your code ends here ########################################################
        ################################################################################
        
        return y_pred  
    