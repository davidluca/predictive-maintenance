# Import Dependencies
# %matplotlib inline

# Start Python Imports
import math, time, random, datetime

# Data Manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

# Machine learning
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool, cv

# Let's be rebels and ignore warnings for now
import warnings
warnings.filterwarnings('ignore')


# Import train & test data
# train = pd.read_csv('../titanic-data/train.csv')
train = pd.read_csv('train.csv')
# test = pd.read_csv('../titanic-data/test.csv')
test = pd.read_csv('test.csv')
# gender_submission = pd.read_csv('../titanic-data/gender_submission.csv') # example of what a submission should look like
gender_submission = pd.read_csv('test_label_sample.csv') # example of what a submission should look like

# View the training data
# print(train.head())
# View the test data (same columns as the training data)
# print(test.head()) # head = view first 5 lines
# View the example submisison dataframe
# print(gender_submission.head())

# Plot graphic of missing values
# print(missingno.matrix(train, figsize = (30,10)))





