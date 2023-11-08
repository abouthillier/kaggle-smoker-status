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

def data_clean(raw_data):

    data = []
    min_max_scaler = preprocessing.MinMaxScaler()

    features_to_scale = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)']
    data = raw_data[features_to_scale]
    data = pd.DataFrame(min_max_scaler.fit_transform(data), columns=(features_to_scale))

    # Find variance in eyesight
    data['eyesight_var'] = abs(raw_data['eyesight(left)'] - raw_data['eyesight(right)'])

    # Find variance in hearing
    data['hearing_var'] = abs(raw_data['hearing(left)'] - raw_data['hearing(right)'])



    return data

smokers = train[train['smoking'] == 1]
non_smokers = train[train['smoking'] == 0]

print(smokers.describe().T)
print(non_smokers.describe().T)

data = train.drop(['smoking'], axis=1)

X = data_clean(data)
y = train['smoking']

print(X.head())

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

# test_data = data_clean(test)

# lr_pred = lr.predict(test_data)
# kn_pred = kn.predict(test_data)
# dt_pred = dtree.predict(test_data)
# grad_pred = grad.predict(test_data)
# svc_pred = svc.predict(test_data)

# model_predictions = pd.DataFrame({'LR': lr_pred,
#                                    'KN': kn_pred,
#                                    'Dtree': dt_pred,
#                                    'grad': grad_pred,
#                                    'svc': svc_pred})

# prediction = meta_model.predict(model_predictions)

# submission = pd.DataFrame({'id':test['id'], 'smoking':prediction})
# submission.to_csv('./submission.csv', index=False)