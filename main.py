import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.linalg import pinv
import csv

df = pd.read_csv('train.csv')

plt.hist(df["SalePrice"])
plt.xlabel("SalePrice")
plt.ylabel("amount")
plt.savefig("continuous fig")

df['SaleType'].value_counts().plot(kind='bar')
plt.xlabel("SaleType")
plt.ylabel("count")
plt.tight_layout()

plt.savefig("discrete fig")

# preprocessing using one hot encoding
# preprocessed HouseStyle as One Hot Encoding, because by commonsense, the style has a
# big impact on the price
df_new = pd.get_dummies(df, columns=["HouseStyle"], prefix="HouseStyle")

# OLS on overal qual, overal cond, and GrLivArea, because these three most likely have a
# linear relation with the price
feature_matrix = np.array([df_new["OverallQual"], df_new["OverallCond"], df_new["GrLivArea"]])
feature_matrix = feature_matrix.T
estimator = pinv(np.dot(feature_matrix.T, feature_matrix))
estimator = np.dot(estimator, feature_matrix.T)
price = np.array(df_new["SalePrice"])
estimator = np.dot(estimator, price)
print(estimator)

#find MSE, R^2
est_price = np.dot(feature_matrix, estimator)
MSE = sum(np.square((est_price - price))) / len(est_price)
print("MSE: " + str(MSE))
R_2 = sum(np.square(est_price - np.mean(price))) / sum(np.square(price - np.mean(price)))
print(R_2)

test_df = pd.read_csv('test.csv')
test_feature = np.array([test_df['OverallQual'], test_df['OverallCond'], test_df['GrLivArea']])
test_feature = test_feature.T
est_price = np.dot(test_feature, estimator)
ID = test_df["Id"]
ID = np.array(ID)
print(est_price)
pd.DataFrame(list(zip(ID, est_price)), columns=["Id","SalePrice"]).to_csv("result.csv", index=False)


pass
