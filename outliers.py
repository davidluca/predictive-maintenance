import seaborn as sns
import pandas as pd


titanic = sns.load_dataset('titanic')
titanic = titanic.copy()
titanic = titanic.dropna()
titanic['age'].plot.hist(
                          bins=50,
                          title="Histogram of the age variable"
                        )

# Z-score
from scipy.stats import zscore
titanic["age_zscore"] = zscore(titanic["age"])
titanic["is_outlier"] = titanic["age_zscore"].apply(
  lambda x: x <= -2.5 or x >= 2.5
)
# print(titanic[titanic["is_outlier"]])



pd.set_option('display.max_rows', 500)
train = pd.read_csv('_train.csv')
from scipy.stats import zscore
train["err1_zscore"] = zscore(train["error2_count"])
train["is_outlier"] = train["err1_zscore"].apply(
  lambda x: x >= 15
)
print(train[train["is_outlier"]])




# # DBSCAN — Density-Based Spatial Clustering of Applications with Noise
# ageAndFare = titanic[["age", "fare"]]
# ageAndFare.plot.scatter(x="age", y="fare")
#
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# ageAndFare = scaler.fit_transform(ageAndFare)
# ageAndFare = pd.DataFrame(ageAndFare, columns=["age", "fare"])
# ageAndFare.plot.scatter(x = "age", y = "fare")
#
# from sklearn.cluster import DBSCAN
# outlier_detection = DBSCAN(eps=0.5,
#                            metric="euclidean",
#                            min_samples=3,
#                            n_jobs=-1)
# clusters = outlier_detection.fit_predict(ageAndFare)
# print(clusters)
#
# from matplotlib import cm
# cmap = cm.get_cmap('Accent')
# ageAndFare.plot.scatter(
#   x = "age",
#   y = "fare",
#   c = clusters,
#   cmap = cmap,
#   colorbar = False
# )
#
#
#
