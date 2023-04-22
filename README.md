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

In AdaBoost, each weak learner is assigned a weight based on its performance in the previous iteration. The amount of say, also known as the weight, given to each weak learner is determined by its ability to correctly classify the training examples.

Specifically, the weight assigned to each weak learner in AdaBoost is proportional to its accuracy in classifying the training examples. The more accurate the weak learner is, the higher its weight will be. This means that the weak learners with higher weights will have a greater influence on the final classification result.

During the boosting process, each weak learner is combined with the previously selected learners to form a strong classifier. The amount of say of each weak learner in the final classification decision is determined by the weights assigned to it during the boosting process. The final classification decision is made by combining the predictions of all the weak learners weighted by their corresponding weights.

The training and testing of the data using the manually made decsion tree and random forest was performed on titanic dataset. The dataset was preprocessed using the Feature engineering steps taken from Sina and Anisotropic. Link(https://www.kaggle.com/code/dmilla/introduction-to-decision-trees-titanic-dataset)

The titanic dataset itself is having slight variations from different sources. The data that I loaded from seaborn was having in columns names that are from kaggle. The data I downloaded from fetch_openml was only having a difference of lower cases for the column names from the kaggle dataset.

