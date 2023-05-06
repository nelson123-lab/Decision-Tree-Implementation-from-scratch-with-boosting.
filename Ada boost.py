# AdaBoost class with the required features.
class AdaBoost:
    def __init__(self, weak_learner, num_learners = 50, learning_rate = 1, max_depth = None, criterion = None):
        """
        weak_learner - The classifier used as a weak learner can be Decision tree or Random Forest in this case.
        num_learners - The maximum number of learners to use when fitting the ensemble.
        learning_rate - The weight applied to each weak learner per iteration.
        estimators - A list to keep track of trained models.
        weights - A list to keep track of weights associated with each model. This will be used during prediction.
        max_depth - The maximum depth of the decision tree
        criterion - The criterion for the decision tree split in the data.
        """
        self.weak_learner = weak_learner
        self.num_learners = num_learners
        self.learning_rate = learning_rate
        self.estimators = []
        self.weights = []
        self.max_depth = max_depth
        self.criterion = criterion

    # Fit Function of the AdaBoost Classifier.
    def fit(self, X, y):
        # Initializing the same weights to all features in the begining of the training.
        # This will be a list of same weights to all the features and will be changing in future according to the importance of feature.
        initial_weights = np.ones(len(X)) / len(X)
        # Iterating num_learners times.
        for i in range(self.num_learners):
            # Training the weak learner on the training set usign the current sample weights.
            classifier = self.weak_learner(max_depth = self.max_depth, criterion = self.criterion)
            # Fitting the classifier along with considering the weights.
            classifier.fit(X, y, weights = initial_weights)
            # Finding the prediction using the weak learner.
            y_pred = classifier.predict(X)
            # Finding the error of the weak learner so that weights of the next learner can be updated.
            error = np.sum(initial_weights * (y != y_pred)) / np.sum(initial_weights)
            # Computing the weights according to the error of the previous weak learner.
            cur_weight = self.learning_rate * np.log((1 - error) / error)
            # Appending the classifier to the estimators list.
            self.estimators.append(classifier)
            # Appending the weights of the corresponding weak learner to the weights list.
            self.weights.append(cur_weight)
            # Updating the initial weights according to the cur_weights form the previous weak learner.
            initial_weights *= np.exp(cur_weight * (y != y_pred))
            initial_weights /= np.sum(initial_weights)
    
    # Function to predict the AdaBoost output.
    def predict(self, X):
        # Making an array of the len of X.
        p_array = np.zeros(len(X))
        # Iterating through each of the weak learner and making the predictions with the weights and corresponding classifier.
        for i, classifier in enumerate(self.estimators):
            y_pred = classifier.predict(X)
            p_array += self.weights[i] * y_pred
        # Taking the sign of the weighted sum to obtain the final prediction of the AdaBoost.
        return np.sign(y_pred)

    # Function to find the Accuracy.
    def Accuracy(self, X_test, y_test):
        # Finding the y_pred inorder to compare with the y_test.
        y_pred = self.predict(X_test)
        # Returing the accuracy of the prediction.
        return round(np.mean(y_pred == y_test)*100, 3)
