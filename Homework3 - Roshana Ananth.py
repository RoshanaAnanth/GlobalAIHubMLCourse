#Homework 3

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score

X, y = make_blobs(n_samples=2000,centers=3, n_features=3,random_state=42)
print(X.shape)
print(y)

#Listing out our features
print("Feature Matrix: ");
print(pd.DataFrame(X, columns=["Feature 1", "Feature 2", "Feature 3"]).head())

#Plotting our dataset
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
#plt.show()
h=pd.DataFrame(X,y, columns=["Feature 1", "Feature 2", "Feature 3"])
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.7, test_size=0.3, random_state=123)

#Decision Tree Model
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train,y_train)
print("Accuracy of train:",clf.score(X_train,y_train))
print("Accuracy of test:",clf.score(X_test,y_test))

plt.figure(figsize=(12, 8))
importance = clf.feature_importances_

sns.barplot(x=importance, y=h.columns)
plt.show()

#Classification report
pred = clf.predict(X_test)
print(classification_report(y_test,pred))

print("Precision = {}".format(precision_score(y_test, pred, average='macro')))
print("Recall = {}".format(recall_score(y_test, pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, pred)))
print("F1 Score = {}".format(f1_score(y_test, pred,average='macro')))
print()
print("The model is neither overfitted nor underfitted. It is perfectly classified")

#XGBoost wouldn't install in my system. Some error kept showing up even after I installed a pre-built binary wheel for Python. So please excuse this. I've tried my best.
