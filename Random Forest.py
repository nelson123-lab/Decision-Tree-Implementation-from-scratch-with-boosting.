# Random forest class according to the requriemnts.
class RandomForest:
    # Initializing the class attributes.
    """
    max_depth: The maximum depth of each decision tree in the random forest.
    criterion: The criterion used for evaluating the quality of splits in each decision tree.
    num_trees: The number of decision trees in the random forest.
    min_features: The minimum number of features to consider when making splits in each decision tree.
    forest: A list of DecisionTree objects.
    """
    def __init__(self, max_depth, criterion, num_trees=100, min_features = 4):
        self.num_trees = num_trees
        self.min_features = min_features
        self.max_depth = max_depth
        self.criterion = criterion
        self.forest = []

    # Defining the fit function that calls the decision tree classifier.
    def fit(self, X, y, weights = None):
        # Finding the number of features in each decision tree.
        n_features = X.shape[1]
        # Iterating through each of the decision trees.
        for i in range(self.num_trees):
            # Bootstrapping
            # 1) Sampling the data with replacement.
            indices = np.random.choice(X.shape[0], size = X.shape[0], replace = True)
            # Making subsets of the data in X_data and y_data.
            X_data, y_data = X[indices], y[indices]
            # 2) Selecting random subset of features
            n_selected_features = np.random.randint(self.min_features, n_features + 1)
            # Selecting the random subset of features and stores in selected_features.
            selected_features = np.random.choice(n_features, size = n_selected_features, replace = False)
            # taking only the selected subset features data.
            X_data = X_data[:, selected_features]
            # Training the Decision Tree 
            Tree = DecisionTree(max_depth = self.max_depth, criterion = self.criterion)
            # Fitting the tree
            Tree.fit(X_data, y_data)
            # Appending the trained decision tree and it's indices of the selected features to the forest list.
            self.forest.append((Tree, selected_features))

    # Function to predict the output of the random forest.
    # Aggregation - Works by maximum voting.
    def predict(self, X):
        # Creating an array to keep track of the votes
        votes_array = np.zeros((X.shape[0],))
        # Iterating through each decision trees and finding the prediction.
        for Tree, selected_features in self.forest:
            prediction = Tree.predict(X[:, selected_features])
            # Updtating the values of the predictions of each decision tree to the votes_array.
            votes_array[prediction == 1] += 1
        # Returing the majority of votes as prediction.
        return (votes_array >= (len(self.forest) / 2)).astype(int)
    
    # Function to find the Accuracy.
    def Accuracy(self, X_test, y_test):
        # Finding the y_pred inorder to compare with the y_test.
        y_pred = self.predict(X_test)
        # Returing the accuracy of the prediction.
        return round(np.mean(y_pred == y_test)*100, 3)
