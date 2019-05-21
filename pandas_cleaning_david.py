from __future__ import print_function

import pandas as pd
pd.__version__

train = pd.read_csv("train_label.csv", sep=",")
# print(train.describe())

train.fillna(0, inplace=True)
train.to_csv('train_out.csv', sep='\t', encoding='utf-8')
