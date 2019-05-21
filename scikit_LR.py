import numpy as np
import pandas as pd

# Load df
# df = np.loadtxt('train_float.csv', delimiter=',', skiprows=1)
df = pd.read_csv('train_float.csv')

# df = df.drop(['error3_count', 'error11_count', 'error19_count',
#                   'error3_max', 'error11_max', 'error19_max',
#                   'error3_min', 'error11_min', 'error19_min',
#                   'error3_mean', 'error11_mean', 'error19_mean',
#                   'error3_std', 'error11_std', 'error19_std'], axis=1, inplace=True)

df.drop([col for col in df.columns if col in ['error3_count', 'error11_count', 'error19_count',
                                              'error3_max', 'error11_max', 'error19_max',
                                              'error3_min', 'error11_min', 'error19_min',
                                              'error3_mean', 'error11_mean', 'error19_mean',
                                              'error3_std', 'error11_std', 'error19_std']], axis=1, inplace=True)



df.fillna(0, inplace=True) #by using - fillna - we can replace NaN with 0

X = df.drop(['breakdown'], axis=1)
y = df.breakdown

import datetime as dt
X['date'] = pd.to_datetime(X['date']) # 24/3/2017 -> 2017-03-24
X['date'] = X['date'].map(dt.datetime.toordinal) # 2015-03-05 -> 735662

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=60, test_size=0.20)
# imparte datele de intrare/iesire in train/test


from sklearn import preprocessing
X_scaled = preprocessing.scale(X_train)
# print(X_scaled)

from sklearn import linear_model
# Fit (train) the Logistic Regression classifier
clf = linear_model.LogisticRegression(C=1e40, solver='newton-cg')
fitted_model = clf.fit(X_train, y_train)

# Predict
prediction_result = clf.predict(X_test)
# print('prediction:\n', prediction_result)
# print('test:\n', y_test)

from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, prediction_result)
# print(roc_auc)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_result)
# print(cm)
