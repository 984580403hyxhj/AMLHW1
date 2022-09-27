import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

# I will pick ticket class, sex. because these are important factors contribute to survival

df = pd.read_csv('train copy.csv').fillna(method = "ffill")
# preprocess one hot on ticket class
df_new = pd.get_dummies(df, columns=["Pclass"], prefix="class")
# preprocess sex with male = 0, female = 1
df_new['Female'] = df_new['Sex'] == 'female'

X = [df_new["class_1"], df_new["class_2"], df_new["class_3"], df_new["Female"]]
X = np.array(X).T
y = df_new["Survived"]
y = np.array(y)

model = LogisticRegression()
model = model.fit(X,y)
print(model.coef_)
test_df = pd.read_csv("test copy.csv")

test_new = pd.get_dummies(test_df, columns=["Pclass"], prefix="class")
# preprocess sex with male = 0, female = 1
test_new['Female'] = test_new['Sex'] == 'female'

X = [test_new["class_1"], test_new["class_2"], test_new["class_3"], test_new["Female"]]
X = np.array(X).T
# y = df_new["Survived"]
# y = np.array(y)
probs = model.predict(X)
print(probs)

ID = test_df["PassengerId"]
ID = np.array(ID)

pd.DataFrame(list(zip(ID, probs)), columns=["PassengerId","Survived"]).to_csv("result2.csv", index=False)

pass




