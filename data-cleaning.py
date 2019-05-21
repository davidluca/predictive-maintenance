import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
# py.init_notebook_mode(connected=True)
# from plotly.offline import init_notebook_mode, iplot
# init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline

import warnings
warnings.filterwarnings('ignore')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
# print(os.listdir("/home/david/Documents/faculta/LICENTA/ml_code"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('train_label.csv')

# print(train.head()) # to check the head of the dataset
# print(train.describe())
# print(train.tail())
# print(train.info())
# print(train.isnull().sum()) # NaN value in the dataset
# print(train[train['label']==1].count()) # 1 value
# print(train[train['label']==0].count()) # 0 value


# data = [go.Scatter(x=train.date, y=train['label'])]
# py.plot(data, filename='time-series-simple')


train.fillna(0, inplace=True) # by using fillna we can replace NaN with 0

# train.info()

X = train.drop('label', axis=1)
y = train.label

# print(X.shape) # returneaza (nr. randuri, nr. coloane)
# print(y.shape)

import datetime as dt
X['date'] = pd.to_datetime(X['date']) # 24/3/2017 -> 2017-03-24
X['date'] = X['date'].map(dt.datetime.toordinal) # 2015-03-05 -> 735662
# print(X['date'].head())

# -------------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=60, test_size=0.20)
# imparte datele de intrare/iesire in train/test

from sklearn import preprocessing
X_scaled = preprocessing.scale(X_train)
print(X_scaled)

from sklearn.tree import DecisionTreeClassifier
decision = DecisionTreeClassifier()
decision.fit(X_train, y_train)
# print(decision.fit(X_train, y_train))

y_predict_decision_tree = decision.predict(X_test)

from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_predict_decision_tree)
# print(roc_auc)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict_decision_tree)
# print(cm)


# from sklearn.metrics import roc_curve
# import matplotlib.pyplot as plt
# fpr, tpr, thresholds = roc_curve(y_test, decision.predict_proba(X_test)[:, 1])
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', label='Decision Tree (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

# ------------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
random = RandomForestClassifier()
random.fit(X_train, y_train)
# print(random.fit(X_train, y_train))

y_predict_random_forest = random.predict(X_test)

from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_predict_random_forest)
# print(roc_auc)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict_random_forest)
# print(cm)

# from sklearn.metrics import roc_curve
# import matplotlib.pyplot as plt
# fpr, tpr, thresholds = roc_curve(y_test, random.predict_proba(X_test)[:, 1])
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', label='Random forest (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()
