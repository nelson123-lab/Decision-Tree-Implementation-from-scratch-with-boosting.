# Decision-Tree-and Random Forest-Implementation-from-scratch-with-boosting.

## Decision tree
The decision tree splits the data into different if else statement with a specific split criteria. If we are dividing the data into different sets based on different features we can have infinite number of spilts. we have to decide on what feature the split should be. This is where the metrics like gini index, entropy, information gain comes into play.

We split the data in a way that it maximizes the information categarized. Decision tree is a recursive algorithm that first finds the information gain according to all the features in the dataset and determines the best features that divides the data set the most.

We now need an algorithm which splits the data according to the maximum information gained feature each time. We make the split tree function for this.

## Random Forest
The random forest algorithm works based on maximum votes from the different decision trees outputs.
Two random process takes place here random feature selection and random sampling with replacement which results in the name random forest. ( Bootstrapping )

The process of finding the maximum votes from the decision trees is called aggregation.

## Adaboost




The training and testing of the data using the manually made decsion tree and random forest was performed on titanic dataset. The dataset was preprocessed using the Feature engineering steps taken from Sina and Anisotropic. Link(https://www.kaggle.com/code/dmilla/introduction-to-decision-trees-titanic-dataset)

The titanic dataset itself is having slight variations from different sources. The data that I loaded from seaborn was having in columns names that are from kaggle. The data I downloaded from fetch_openml was only having a difference of lower cases for the column names from the kaggle dataset.
