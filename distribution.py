import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
# we don't like warnings
# you can comment the following 2 lines if you'd like to
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('train_float.csv')

pd.set_option('display.max_rows', 500)


train = pd.read_csv('_train.csv')
test = pd.read_csv('_test.csv')


df_bin = pd.DataFrame()  # for discretised continuous variables
df_con = pd.DataFrame()  # for continuous variables


def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):
    """
    Function to plot counts and distributions of a label variable and
    target variable side by side.
    ::param_data:: = target dataframe
    ::param_bin_df:: = binned dataframe for countplot
    ::param_label_column:: = binary labelled column
    ::param_target_column:: = column you want to view counts and distributions
    ::param_figsize:: = size of figure (width, height)
    ::param_use_bin_df:: = whether or not to use the bin_df, default False
    """
    if use_bin_df:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=bin_df);
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column],
                     kde_kws={"label": "broke"});
        sns.distplot(data.loc[data[label_column] == 0][target_column],
                     kde_kws={"label": "not broke"});
    else:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=data);
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column],
                     kde_kws={"label": "broke"});
        sns.distplot(data.loc[data[label_column] == 0][target_column],
                     kde_kws={"label": "not broke"});


# Let's add this to our subset dataframes
df_bin['breakdown'] = train['breakdown']
df_con['breakdown'] = train['breakdown']

df_bin['error1_count'] = train['error1_count']
df_con['error1_count'] = train['error1_count']


# Visualise the counts of error1_count and the distribution of the values against breakdown
plot_count_dist(train,
                bin_df=df_bin,
                label_column='breakdown',
                target_column='error1_count',
                figsize=(20, 10))


