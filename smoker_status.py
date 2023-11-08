# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import re

pd.options.display.max_columns = None

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# Define a function to extract the numeric part from the first item
def extract_and_convert(value):
    if isinstance(value, str):
        parts = value.split(' ')  # Split the value by spaces
        numeric_part = ''.join(filter(str.isdigit, parts[0]))
        return int(numeric_part) if numeric_part else 0
    return 0

def count_cabins(cell):
    if pd.notna(cell):
        return len(cell.split())
    else:
        return 1

def cabin_convert(cell, cabin_dict):
    if pd.notna(cell):
        parts = cell.split(' ')
        alpha = ''.join(filter(str.isalpha, parts[0]))
        if alpha in cabin_dict:
            return cabin_dict[alpha]
        else:
            return 0
    else: 
        return 0

def data_clean(raw_data):

    min_max_scaler = preprocessing.MinMaxScaler()
    scaler = preprocessing.StandardScaler()

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare_per_room', 'Embarked']

    # Fill missing Age data with the median
    raw_data['Age'] = raw_data['Age'].fillna(raw_data['Age'].median())
    raw_data['Age'] = min_max_scaler.fit_transform(raw_data[['Age']])

    # Fill missing Fare data with $0
    raw_data['Fare'] = raw_data['Fare'].fillna(0)
    
    # Calculate per-room fare
    raw_data['Fare_per_room'] = np.trunc((raw_data['Fare'] / raw_data['Cabin'].apply(count_cabins)))
    raw_data['Fare_per_room'] = min_max_scaler.fit_transform(raw_data[['Fare_per_room']])

    # Convert Cabin values to categorized columns
    cabins = 'ABCDEFG'

    for letter in cabins:
        cabin_label = f'Cabin_{letter}'
        raw_data[cabin_label] = np.where(raw_data['Cabin'].astype(str).str.contains(letter), raw_data['Cabin'], 0)
        raw_data[cabin_label] = raw_data[cabin_label].apply(extract_and_convert)

    if 'Survived' in raw_data.columns:
        surv = raw_data[raw_data['Survived'] == 1]
        cabin_percent = []
        for letter in cabins:
            cabin_label = f'Cabin_{letter}'
            # Calculate the percent of survivors in each cabin deck
            percent = np.count_nonzero(surv[cabin_label]) / np.count_nonzero(raw_data[cabin_label])  
            cabin_percent.append("%.2f" % percent)
    else:
        cabin_percent = [0.47, 0.74, 0.59, 0.77, 0.75, 0.78, 0.50]

    # Convert Sex to binary
    sex = {'female': 1, 'male': 0}
    raw_data['Sex'] = raw_data['Sex'].map(sex)

    # Flip values of Passenger Class and scale exponentially
    # Lower Passenger Class indicates higher economic class, 1=upper, 3=lower
    # New scale 0=lower, 2.71=mid, 7.38=upper
    raw_data['Pclass'] = np.exp([3 - value for value in raw_data['Pclass']])
   
    data = pd.get_dummies(raw_data[features])

    # Create a dictionary of Cabin letter with percent liklihood of survival
    cabin_dict = dict(zip(list(cabins), cabin_percent))

    # Convert cabins to a percent liklihood of servival
    data['Cabin'] = raw_data['Cabin'].apply(cabin_convert, args=(cabin_dict,))

    return data

# data = data_clean(train)
print(train.head(20))

X = train
y = train['smoking']


# Split the data into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

kn = KNeighborsClassifier(n_neighbors = 2)
kn.fit(X_train, y_train)
kn_pred = kn.predict(X_test)

dtree = tree.DecisionTreeClassifier(criterion='gini')
dtree.fit(X_train, y_train)
dt_pred = dtree.predict(X_test)

# Create a Gradient Boosting Classifier
grad = GradientBoostingClassifier(n_estimators=100,
                                max_depth=4,
                                random_state=42)
grad.fit(X_train, y_train)
grad_pred = grad.predict(X_test)

# Define individual classifiers
svc = SVC(kernel='linear', C=1)
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)

base_model_predictions = pd.DataFrame({'LR': lr_pred,
                                       'KN': kn_pred,
                                       'Dtree': dt_pred,
                                       'grad': grad_pred,
                                       'svc': svc_pred})

meta_model = tree.DecisionTreeClassifier(criterion='gini')
meta_model.fit(base_model_predictions, y_test)
y_pred = meta_model.predict(base_model_predictions)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

test_data = data_clean(test)

lr_pred = lr.predict(test_data)
kn_pred = kn.predict(test_data)
dt_pred = dtree.predict(test_data)
grad_pred = grad.predict(test_data)
svc_pred = svc.predict(test_data)

model_predictions = pd.DataFrame({'LR': lr_pred,
                                   'KN': kn_pred,
                                   'Dtree': dt_pred,
                                   'grad': grad_pred,
                                   'svc': svc_pred})

prediction = meta_model.predict(model_predictions)

submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Transported':prediction})
submission.to_csv('./submission.csv', index=False)