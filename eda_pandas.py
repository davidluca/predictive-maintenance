import numpy as np
import pandas as pd
# we don't like warnings
# you can comment the following 2 lines if you'd like to
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('train_float.csv')

pd.set_option('display.max_rows', 500)

# print(df.head())
# print(df.shape)
# print(df.info())

# columns_name = list(df.columns.values)
# columns_name.remove('breakdown')
# columns_name.remove('date')
# print(columns_name)

# df[columns_name] = df[columns_name].astype(float)  #convert int to float
# print(list(df.columns.values))
# df.to_csv('train_float.csv')

# print(df.describe())

'''Boolean indexing with one column is also very convenient. The syntax is df[P(df['Name'])], 
where P is some logical condition that is checked for each element of the Name column. 
The result of such indexing is the DataFrame consisting only of rows that satisfy the P condition on the Name column.

Let’s use it to answer the question:

What are average values of numerical features for churned users?'''

print(df[df['breakdown'] == 1].mean())  # media valorilor de pe randurile unde breakdown == 1

# print(df.sum())  # suma valorilor per coloane
# print(df['error3_count'].sum())

# print(df['breakdown'].mean())  # media nr de masini stricate

'''DataFrames can be indexed by column name (label) or row name (index) or by the serial number of a row.
 The loc method is used for indexing by name, while iloc() is used for indexing by number.
In the first case below, we say "give us the values of the rows with index from 0 to 5 (inclusive) and columns labeled from State to Area code (inclusive)".
 In the second case, we say "give us the values of the first five rows in the first three columns" (as in a typical Python slice: the maximal value is not included).'''
# print(df.loc[0:5, 'breakdown':'error3_count'])
# print(df.iloc[0:5, 0:3])


# If we need the first or the last line of the data frame, we can use the df[:1] or df[-1:] construct:


'''To apply functions to each column, use apply():'''
# print(df.apply(np.max))
'''The apply method can also be used to apply a function to each row. 
To do this, specify axis=1. Lambda functions are very convenient in such scenarios. For example, if we need to select all states starting with W, we can do it like this:'''
# print(df[df['State'].apply(lambda state: state[0] == 'W')].head())
# print(df[df['breakdown'].apply(lambda value: value == 1)].head())


'''The map method can be used to replace values in a column by passing a dictionary of the form {old_value: new_value} as its argument:'''
# d = {'No' : False, 'Yes' : True}
# df['International plan'] = df['International plan'].map(d)
# df.head()
'''The same thing can be done with the replace method:'''
# df = df.replace({'Voice mail plan': d})
# df.head()


'''
   df.groupby(by=grouping_columns)[columns_to_show].function()
1. First, the groupby method divides the grouping_columns by their values. They become a new index in the resulting dataframe.
2. Then, columns of interest are selected (columns_to_show). If columns_to_show is not included, all non groupby clauses will be included.
3. Finally, one or several functions are applied to the obtained groups per selected columns.
Here is an example where we group the data according to the values of the Churn variable and display statistics of three columns in each group:
'''
# columns_to_show = ['error1_count', 'error2_min', 'error3_max'] #  list(df) - toate feature-urile
# print(df.groupby(['breakdown'])[columns_to_show].describe(percentiles=[]))

'''Let’s do the same thing, but slightly differently by passing a list of functions to agg():'''
# df.groupby(['breakdown'])[columns_to_show].agg([np.mean, np.std, np.min, np.max])


'''Suppose we want to see how the observations in our sample are distributed in the context of two variables -  
Churn and International plan. To do so, we can build a contingency table using the crosstab method:'''
# print(pd.crosstab(df['breakdown'], df['error1_count']))
# print(pd.crosstab(df['breakdown'], df['error1_count'], normalize=True))
# print(pd.crosstab(df['breakdown'], df['error1_count'], margins=True))


'''For example, if we want to calculate the total number of calls for all users, 
let’s create the total_calls Series and paste it into the DataFrame:'''
# total_calls = df['error1_count'] + df['error2_count'] + \
#               df['error3_count'] + df['error4_count']
# df.insert(loc=len(df.columns), column='Sum of errors', value=total_calls)
''' loc parameter is the number of columns after which to insert the Series object
 we set it to len(df.columns) to paste it at the very end of the dataframe'''
# print(df.head())


'''To delete columns or rows, use the drop method, passing the required indexes and the axis parameter 
(1 if you delete columns, and nothing or 0 if you delete rows). The inplace argument tells whether to change the original DataFrame. 
With inplace=False, the drop method doesn't change the existing DataFrame and returns a new one with dropped rows or columns. 
With inplace=True, it alters the DataFrame.'''
'''get rid of just created columns'''
# print(df.head())
# df.drop(['error3_count'], axis=1, inplace=True)
# print(df.head())
'''and here’s how   you can delete rows'''
# df.drop([1, 2]).head()



'''some imports to set up plotting'''
import matplotlib.pyplot as plt
import seaborn as sns

# sns.countplot(x='error1_count', hue='breakdown', data=df)

# df['er1_count<3'] = (df['error1_count'] < 4).astype('float')
# print(pd.crosstab(df['er1_count<3'], df['breakdown'], margins=True))
