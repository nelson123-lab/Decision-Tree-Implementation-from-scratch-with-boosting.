class RandomForest:
    def __init__(self, max_depth,criterion,num_trees=100,min_features =4):
        #self.classifier = classifier
        self.num_trees = num_trees
        self.min_features = min_features
        self.max_depth=max_depth
        self.criterion=criterion
        self.forest = []

    def fit(self, X, y):
        num_features = X.shape[1]
        for i in range(self.num_trees):
            # Sampling with replacement
            indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            # Selecting random subset of features
            num_selected_features = np.random.randint(self.min_features, num_features + 1)
            selected_features = np.random.choice(num_features, size=num_selected_features, replace=False)
            X_sample = X_sample[:, selected_features]
            # Train decision tree
            clf = DecisionTreeClassifier(max_depth=self.max_depth,criterion=self.criterion)
            clf.fit(X_sample, y_sample)
            self.forest.append((clf, selected_features))

    def predict(self, X):
        votes = np.zeros((X.shape[0],))
        for clf, selected_features in self.forest:
            X_sample = X[:, selected_features]
            pred = clf.predict(X_sample)
            votes[pred == 1] += 1
        return (votes >= (len(self.forest) / 2)).astype(int)
    
    def Accuray(self, X_test, y_test):
       y_pred = self.predict(X_test)
       return np.mean(y_pred == y_test)
