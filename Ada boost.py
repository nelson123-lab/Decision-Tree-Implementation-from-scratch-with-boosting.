class AdaBoost:
    def __init__(self, base_estimator, n_estimators=50, learning_rate=1, max_depth = None, criterion = None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.weights = []
        self.max_depth = max_depth
        self.criterion = criterion
        
    def fit(self, X, y):
        # Initialize sample weights
        sample_weight = np.ones(len(X)) / len(X)
        
        for i in range(self.n_estimators):
            # Train a weak learner on the training set using the current sample weights
            estimator = self.base_estimator(max_depth = self.max_depth, criterion = self.criterion)
            estimator.fit(X, y)
            
            # Compute the error of the weak learner on the training set
            y_pred = estimator.predict(X)
            error = np.sum(sample_weight * (y != y_pred)) / np.sum(sample_weight)
            
            # Compute the weight of the weak learner in the final classifier
            weight = self.learning_rate * np.log((1 - error) / error)
            self.weights.append(weight)
            self.estimators.append(estimator)
            
            # Update the sample weights based on the performance of the weak learner
            sample_weight *= np.exp(weight * (y != y_pred))
            sample_weight /= np.sum(sample_weight)
    
    def predict(self, X):
        # Combine the weak learners into a final strong classifier
        y_pred = np.zeros(len(X))
        for i in range(len(self.estimators)):
            y_pred += self.weights[i] * self.estimators[i].predict(X)
        
        # Make predictions on the testing set using the final classifier
        return np.sign(y_pred)

    def Accuray(self, X_test, y_test):
       y_pred = self.predict(X_test)
       return np.mean(y_pred == y_test)
