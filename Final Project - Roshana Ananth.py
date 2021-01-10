#Final Project
#Note : There are a few plots used for data analysis. Once done with viewing the plot, please close the plot window alone and the rest of the program will continue to execute.


from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score, recall_score,f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Reading our dataset winequality.csv
data = pd.read_csv("https://raw.githubusercontent.com/globalaihub/introduction-to-machine-learning/main/Final%20Project/winequality.csv")
#Analysing our data
print()
print(data.head())
print("\nDataset Description")
print(data.describe())
print("\nChecking the correlation of our features")
print(data.corr())
print("\nGetting more information about datatypes")
print(data.info())

#Exploratory Data Analysis with different types of plots
sns.countplot(x='quality', data=data)
sns.displot(x='quality',data=data)
sns.boxplot('quality', 'alcohol', data = data)
plt.show()

#Now we will create a list called Reviews. If reviews=1(1,2,3), then it is bad. If it is 2(4,5,6,7),it is average and 3(8,9,10) means it is excellent.

reviews = []
for i in data['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
data['Reviews'] = reviews
print("\nThe Wine Reviews - 1:Bad  2:Average   3:Excellent")
print(Counter(data['Reviews']))

#Splitting the variables
x = data.iloc[:,:11]
y = data['Reviews']

#Scaling the data
sc = StandardScaler()
x = sc.fit_transform(x)
#print(x)

#Principal Component Analysis (PCA)
pca = PCA()
x_pca = pca.fit_transform(x)
#Plotting the graph
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()
plt.show()
#We will pick 8 principal components that cause the most variation in our graph.
pca_new = PCA(n_components=8)
x_new = pca_new.fit_transform(x)

#Splitting our dataset into train and test data
x_train, x_test, y_train, y_test = train_test_split(x_new, y,train_size=0.7, test_size = 0.3,random_state=42)

#Logistic Regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_predict = lr.predict(x_test)
#printing confusion matrix and accuracy score for the model
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)
print("\nLogistic Regression Model")
print("\nThe Confusion Matrix for Logistic Regression Model")
print(lr_conf_matrix)
print("\nThe Accuracy Score")
print(lr_acc_score*100)
print("Precision = {}".format(precision_score(y_test, lr_predict, average='macro',zero_division=1)))
print("Recall = {}".format(recall_score(y_test, lr_predict, average='macro')))
print("F1 Score = {}".format(f1_score(y_test, lr_predict,average='macro')))

#Decision tree Model
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_predict = dt.predict(x_test)
dt_conf_matrix = confusion_matrix(y_test, dt_predict)
dt_acc_score = accuracy_score(y_test, dt_predict)
print("\nDecision Tree Model")
print("\nThe Confusion Matrix for Decision Tree Model")
print(dt_conf_matrix)
print("\nThe Accuracy Score")
print(dt_acc_score*100)
print("Precision = {}".format(precision_score(y_test, dt_predict, average='macro',zero_division=1)))
print("Recall = {}".format(recall_score(y_test, dt_predict, average='macro')))
print("F1 Score = {}".format(f1_score(y_test, dt_predict,average='macro')))

#Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_predict=rf.predict(x_test)
rf_conf_matrix = confusion_matrix(y_test, rf_predict)
rf_acc_score = accuracy_score(y_test, rf_predict)
print("\nRandom Forest Classifier")
print("\nThe Confusion Matrix for Random Forest Classifier Model")
print(rf_conf_matrix)
print("\nThe Accuracy Score")
print(rf_acc_score*100)
print("Precision = {}".format(precision_score(y_test, rf_predict, average='macro',zero_division=1)))
print("Recall = {}".format(recall_score(y_test, rf_predict, average='macro')))
print("F1 Score = {}".format(f1_score(y_test, rf_predict,average='macro')))

#Concluding
print("\nAs we can see from the model scores, the Logistic Regression model and the Random Forest Classifier perform equally well with 98.5% whereas our Decision Tree model has lesser accuracy.")
print("\n We can conclude that Logistic Regression Model performs exceptionally well, not just in the accuracy score but also in the precision score")
