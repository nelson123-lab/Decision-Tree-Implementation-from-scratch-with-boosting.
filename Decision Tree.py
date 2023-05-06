# Decision Tree class according to the given requirements.
class DecisionTree:
    """
    criterion - Either misclassification rate, Gini I, or entropy.
    max_depth - The maximum depth the tree should grow.
    min_samples_split - The minimum number of samples required to split.
    min_samples_leaf - The minimum number of samples required for a leaf node.
    """
    # Initializing the above parameters.
    def __init__(self, criterion = 'gini', max_depth = None, min_samples_split = 2, min_samples_leaf = 1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
    
    # Writing the functions for different criterion used in the decision tree for splitting the data based on priority.
    # Gini I
    def gini(self, y):
        # Finds the probabilities of each class and divides it with the total number of samples.
        probabilities = np.bincount(y) / len(y)
        # Returns the gini I of the classification.
        return 1 - np.sum(probabilities ** 2)

    # Misclassification rate    
    def misclassification_rate(self, y):
        # Finding the counts of each class and number of samples.
        counts, n_samples = np.bincount(y), len(y)
        # Returing zero if the number of samples is zero.
        if n_samples == 0:  return 0
        # Returing the misclassfication rate.
        return 1 - (np.max(counts) / n_samples)
    
    # Function for entropy
    def entropy(self, y):
        # Finding the number of occurances of each of the class and probabilities of the of each class.
        probabilities = np.bincount(y) / len(y)
        # Returning the entropy of the classification.
        return -sum(p * np.log2(p) for p in probabilities if p > 0)
        
    # Function to fit the tree
    def fit(self, X, y, weights = None):
        # Finding the unique classes in the target data.
        self.n_classes = len(np.unique(y))
        # Method of the Decision Tree to recursively grow the tree.
        self.tree = self.Tree(X, y, depth = 0)

    # Function to predict the data.  
    def predict(self, X):
        # Returning the output of the traversed tree.
        return np.array([self.traversal(x, self.tree) for x in X])
        
    def Tree(self, X, y, depth = 0):
        # Finding the number of classes and the number of features in the independent features.
        n_samples, n_features = X.shape
        # Finding the unique classes in the target data.
        n_labels = len(np.unique(y))
        
        # Stopping criteria for the splitting the data into right and left nodes.
        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1 or n_samples < self.min_samples_split:
            return {'p_class': np.bincount(y).argmax()}
        
        # Taking the criterion from the input and applying it to the different functions accordingly.
        # Gini I if criterion is 'gini'
        if self.criterion == 'gini': criterion = self.gini
        # Misclassification rate if criterion is 'misclassification_rate'
        elif self.criterion == 'misclassification_rate': criterion = self.misclassification_rate
        # Entropy if criterion is 'entropy'
        elif self.criterion == 'entropy': criterion = self.entropy
        # Else the the Criterion is not specified while creating the decision tree object.
        else: raise ValueError('No criterion specified')
        
        # Stores the indexes of the features as a list in the feature index
        feature_index = np.arange(n_features)
        # Checking if there are more than one feature in the dataset.
        # This is useful when the several number of splits has already happened.
        if n_features > 1:
            # Randomely shuffling the array.
            np.random.shuffle(feature_index)
            # Selecting a subset of the feature indices where the size of the subset is between 1 and n_features.
            feature_index = feature_index[:np.random.randint(1, n_features)]
        """
        Below is the part of the decision tree where we find the best feature and threshold to split the node.
        # thresholds are the unique values of the feature in the dataset X.
        """
        # Initializing the impurity to infinity so that the current impurity can be compared and later changed accordingly.
        I = np.inf
        # Iterating through the features in the dataset.
        for feature in feature_index:
            # Finding the thresholds to split the data on the given feature.
            thresholds = np.unique(X[:, feature])
            # Iterates over each possible threshold of the current feature from the feature index choosen.
            for threshold in thresholds:
                # Splitting the dataset into left and right child.
                left_child, right_child = y[X[:, feature] < threshold], y[X[:, feature] >= threshold]
                # Checking whether the split meets the condition of min_samples_leaf on both the sides.
                if len(left_child) < self.min_samples_leaf or len(right_child) < self.min_samples_leaf: continue
                # Finding the impurity of the current split depending on the criterion used from 'gini', 'missclassification_rate', 'Entropy'.
                curr_I = (len(left_child)/n_samples) * criterion(left_child) + (len(right_child)/n_samples) * criterion(right_child)
                # Comparing the current impurity vs the initial impurity.
                if curr_I < I:
                    # Changing the current impurity to the initial impurity, best feature as the feature and best threshold as threshold.
                    I, b_f, b_t = curr_I, feature, threshold
                    # Computing the indices of the samples in X that belong to the left and right subsets.
                    left_child_index, right_child_index = X[:, feature] < threshold, X[:, feature] >= threshold
                    
        # Recursive implementation of the decision tree algorithm.
        if I == np.inf: return {'p_class': np.bincount(y).argmax()}
        # Returing a dictionary containing predicted class for the current node.
        left_data, right_data = self.Tree(X[left_child_index], y[left_child_index], depth + 1), self.Tree(X[right_child_index], y[right_child_index], depth + 1)
        # Returining a dictionary with the feature and threshold that produced the lowest impurity in the current node.
        return {'feature': b_f, 'threshold': b_t, 'left': left_data, 'right': right_data}
    
    # Function for traversal in the tree.(Recursively traverse through the tree.)
    def traversal(self, x, node):
        # checking if the current node is a leaf node.
        if 'p_class' in node:   return node['p_class']
        # checking if the value of the feature at the current node is less than the threshodl value.
        if x[node['feature']] < node['threshold']:  return self.traversal(x, node['left'])
        # Checking if the value of the feature at the current node is greater than or equal to the threshold value.
        else:   return self.traversal(x, node['right'])

    # Function to find the Accuracy.
    def Accuracy(self, X_test, y_test):
        # Finding the y_pred inorder to compare with the y_test.
        y_pred = self.predict(X_test)
        # Returing the accuracy of the prediction.
        return round(np.mean(y_pred == y_test)*100, 3)
    
# Testing
# Checking the accuracy with the Criterion as Gini Impurity.
Tree = DecisionTree(criterion='gini', max_depth = 5, min_samples_split = 10, min_samples_leaf = 2)
Tree.fit(X_train.values, y_train.values)
Accuracy = Tree.Accuracy(X_test.values, y_test.values)
print('Accuracy',Accuracy)
