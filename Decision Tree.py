class DecisionTreeClassifier:
    def __init__(self, max_depth=None, criterion='gini', min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_classes = len(np.unique(y))
      
        self.tree = self._grow_tree(X, y, depth=0)
        
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1 or n_samples < self.min_samples_split:
            return {'predicted_class': np.bincount(y).argmax()}
        
        if self.criterion == 'gini':
            impurity_func = self._gini
        elif self.criterion == 'misclassification_rate':
            impurity_func = self._misclassification_rate
        elif self.criterion == 'entropy':
            impurity_func = self._entropy
        else:
            raise ValueError('Invalid criterion')
        
        feature_idxs = np.arange(n_features)
        if n_features > 1:
            np.random.shuffle(feature_idxs)
            feature_idxs = feature_idxs[:np.random.randint(1, n_features)]
            
        impurity = np.inf
        for feature in feature_idxs:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                y_left = y[X[:, feature] < threshold]
                y_right = y[X[:, feature] >= threshold]
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                curr_impurity = (len(y_left)/n_samples) * impurity_func(y_left) + (len(y_right)/n_samples) * impurity_func(y_right)
                if curr_impurity < impurity:
                    impurity = curr_impurity
                    best_feature = feature
                    best_threshold = threshold
                    left_idxs = X[:, feature] < threshold
                    right_idxs = X[:, feature] >= threshold
                    
        if impurity == np.inf:
            return {'predicted_class': np.bincount(y).argmax()}
        
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth+1)
        
        return {'feature': best_feature, 'threshold': best_threshold, 'left': left, 'right': right}

    
    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        impurity = 1.0
        for count in counts:
            prob = count / len(y)
            impurity -= prob ** 2
        return impurity
    
    def _misclassification_rate(self, y):
        _, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        if n_samples == 0:
            return 0
        max_count = np.max(counts)
        return 1 - (max_count / n_samples)
    
    def _entropy(self, y):
      _, counts = np.unique(y, return_counts=True)
      probs = counts / len(y)
      entropy = 0
      for p in probs:
          if p > 0:
              entropy -= p * np.log2(p)
      return entropy
     
    def _traverse_tree(self, x, node):
        if 'predicted_class' in node:
            return node['predicted_class']
        
        if x[node['feature']] < node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])

    def Accuray(self, X_test, y_test):
       y_pred = self.predict(X_test)
       return np.mean(y_pred == y_test)
