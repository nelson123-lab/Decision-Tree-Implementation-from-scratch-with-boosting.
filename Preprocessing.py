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

# Converting the Targets from strings to integers.
y_train, y_test = y_train.astype(int), y_test.astype(int)

# Preprocessing method given by Sina and Anisotropic
full_data = [X_train, X_test]

# Updating NaN values by 0 and others by 1.
X_train['Has_Cabin'] = X_train["cabin"].apply(lambda x: 0 if type(x) == float else 1)
X_test['Has_Cabin'] = X_test["cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Creating new feature FamilySize as a combination of SibSp and Parch
# SibSp - Number of siblings or spouses a passenger had on the RMS Titanic.
# Parch -  Number of parents or kids a passenger brought aboard the RMS Titanic.
for dataset in full_data: # To loop through both train and test.
    dataset['FamilySize'] = dataset['sibsp'] + dataset['parch'] + 1

# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['embarked'] = dataset['embarked'].fillna('S')
# Remove all NULLS in the Fare column
for dataset in full_data:
    dataset['fare'] = dataset['fare'].fillna(X_train['fare'].median())

# Remove all NULLS in the Age column
for dataset in full_data:
    age_avg = dataset['age'].mean()
    age_std = dataset['age'].std()
    age_null_count = dataset['age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    # Next line has been improved to avoid warning
    dataset.loc[np.isnan(dataset['age']), 'age'] = age_null_random_list
    dataset['age'] = dataset['age'].astype(int)

# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex
    dataset['sex'] = dataset['sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked
    dataset['embarked'] = dataset['embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['fare'] <= 7.91, 'fare'] = 0
    dataset.loc[(dataset['fare'] > 7.91) & (dataset['fare'] <= 14.454), 'fare'] = 1
    dataset.loc[(dataset['fare'] > 14.454) & (dataset['fare'] <= 31), 'fare']   = 2
    dataset.loc[ dataset['fare'] > 31, 'fare'] 							        = 3
    dataset['fare'] = dataset['fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['age'] <= 16, 'age'] 					       = 0
    dataset.loc[(dataset['age'] > 16) & (dataset['age'] <= 32), 'age'] = 1
    dataset.loc[(dataset['age'] > 32) & (dataset['age'] <= 48), 'age'] = 2
    dataset.loc[(dataset['age'] > 48) & (dataset['age'] <= 64), 'age'] = 3
    dataset.loc[ dataset['age'] > 64, 'age']
    
# Dropping the features that are no longer required.
drop_elements = ['name', 'ticket', 'cabin', 'sibsp']
X_train = X_train.drop(drop_elements, axis = 1)
X_test  = X_test.drop(drop_elements, axis = 1)
