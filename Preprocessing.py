# Importing libraries
import numpy as np
import regex as re
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# as_frame = True -- To return as a pandas dataframe, return_X_y = True -- To return the (X) Independent variable and (y) target separately.
X, y = fetch_openml("titanic", version = 1, as_frame = True, return_X_y = True) 

"""
Removing the features that are not useful in predicting the survival of the passengers.
Information regarding the lifeboat number that a passenger was given is listed in the 'boat' column.
The body identification number of each passenger is listed in the 'body' column.
The passenger's origin and destination are listed in the 'home.dest' column.
"""
X.drop(['boat', 'body', 'home.dest'], axis=1, inplace=True)

# Splitting the data using train_test_split.
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2)


