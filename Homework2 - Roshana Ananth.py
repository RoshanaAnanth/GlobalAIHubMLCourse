#Homework 2
#The code that I've commented are the code that I used to analyse the data. The only lines that are printed are the scores of each model.


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import preprocessing
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#loading data from the boston dataset
Xb,yb =load_boston(return_X_y=True)

df_boston = pd.DataFrame(Xb,columns = load_boston().feature_names)
#print(df_boston.head())  #returns the first 5 rows from our dataset
#print(df_boston.info())  #shows information about the number of entries, columns and datatype
#print(df_boston.describe())  # shows some details of our dataset in a tabulated manner

#print(df_boston.isna().sum())  #checks for any na values in our data

#print(df_boston.corr)  #shows us the correlation of different features in our dataset


'''
#new_df=new_df.cumsum()
plt.figure()
new_df.plot()
plt.show()

corr = new_df.corr()

plt.figure(figsize=(14, 14))
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True, annot = True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
ax.set_ylim(len(corr)+0.5, -0.5);
plt.show()
'''

def adj_r2 (X,y,model):
    r_squared = model.score(X,y)
    return(1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1))

#print(df_boston.isin(['?']).sum())

#Linear regression
X_train, X_test, y_train, y_test = train_test_split(Xb,yb,train_size=0.7, test_size=0.3, random_state=42)

modelb = LinearRegression(normalize=False)
modelb.fit(X_train,y_train)
print("\n Linear Regression")
print("Score of the train set",modelb.score(X_train,y_train))
print("Score of the test set",modelb.score(X_test,y_test))
print("Adj. R2 of the train set",adj_r2(X_train,y_train,modelb))
print("Adj. R2 of the test set",adj_r2(X_test,y_test,modelb))
print()
#Checking the importance of each feature to see if we need to drop any feature 
'''importance = modelb.coef_
for i in range(len(importance)):
    print("Feature", df_boston.columns[i], "Score:", importance[i])
'''
#Dropping features that decrease our performance
new_df = df_boston.drop(["INDUS","AGE"],axis=1)
print("\n After dropping INDUS and AGE")
X_train, X_test, y_train, y_test = train_test_split(new_df,yb,train_size=0.7, test_size=0.3, random_state=42)
modelb = LinearRegression(normalize=False)
modelb.fit(X_train,y_train)
print("Score of the train set",modelb.score(X_train,y_train))
print("Score of the test set",modelb.score(X_test,y_test))

#Checking for outliers and removing them
z=np.abs(stats.zscore(new_df))
outliers=list(set(np.where(z>3)[0]))
new_df1=new_df.drop(outliers,axis=0).reset_index(drop=False)

y_new = yb[list(new_df1["index"])]
X_new = new_df1.drop('index', axis = 1)
X_scaled = StandardScaler().fit_transform(X_new)

print("\n After taking out the outliers")
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y_new,train_size=0.7, test_size=0.3, random_state=42)
modelb = LinearRegression(normalize=False)
modelb.fit(X_train,y_train)
print("Score of the train set",modelb.score(X_train,y_train))
print("Score of the test set",modelb.score(X_test,y_test))

#Trying out regression models with different alpha values
X_train, X_test, y_train,y_test = train_test_split(X_scaled, y_new, train_size=0.7, test_size = 0.3, random_state = 102)

print("\n Ridge regression with alpha = 0.002")
ridge_model = Ridge(alpha = 0.002)
ridge_model.fit(X_train, y_train)
print("Score of the train set",ridge_model.score(X_train,y_train))
print("Score of the test set",ridge_model.score(X_test,y_test))

print("\n Ridge regression with alpha = 0.0001")
ridge_model = Ridge(alpha = 0.0001)
ridge_model.fit(X_train, y_train)
print("Score of the train set",ridge_model.score(X_train,y_train))
print("Score of the test set",ridge_model.score(X_test,y_test))

print("\n Ridge regression with alpha = 0.1")
ridge_model = Ridge(alpha = 0.1)
ridge_model.fit(X_train, y_train)
print("Score of the train set",ridge_model.score(X_train,y_train))
print("Score of the test set",ridge_model.score(X_test,y_test))

print("\n Ridge regression with alpha = 0.9")
ridge_model = Ridge(alpha = 0.9)
ridge_model.fit(X_train, y_train)
print("Score of the train set",ridge_model.score(X_train,y_train))
print("Score of the test set",ridge_model.score(X_test,y_test))

print("\n Ridge regression with alpha = 3.0")
ridge_model = Ridge(alpha = 2.0)
ridge_model.fit(X_train, y_train)
print("Score of the train set",ridge_model.score(X_train,y_train))
print("Score of the test set",ridge_model.score(X_test,y_test))

print("\n Lasso Regularization with alpha = 0.0001")
lasso_model = Lasso(alpha = 0.0001)
lasso_model.fit(X_train, y_train)
print("Score of the train set",lasso_model.score(X_train,y_train))
print("Score of the test set",lasso_model.score(X_test,y_test))

print("\n Lasso Regularization with alpha = 0.0004")
lasso_model = Lasso(alpha = 0.0004)
lasso_model.fit(X_train, y_train)
print("Score of the train set",lasso_model.score(X_train,y_train))
print("Score of the test set",lasso_model.score(X_test,y_test))

print("\n Lasso Regularization with alpha = 0.01")
lasso_model = Lasso(alpha = 0.01)
lasso_model.fit(X_train, y_train)
print("Score of the train set",lasso_model.score(X_train,y_train))
print("Score of the test set",lasso_model.score(X_test,y_test))

print("\n Lasso Regularization with alpha = 0.1")
lasso_model = Lasso(alpha = 0.1)
lasso_model.fit(X_train, y_train)
print("Score of the train set",lasso_model.score(X_train,y_train))
print("Score of the test set",lasso_model.score(X_test,y_test))

print("\nBoth Ridge regression and Lasso regularization perform equally well for values 0.0001. Comparing to our normal Linear Regression, the score of the test set is increased in Lasso and ridge regression, whereas the score of the train set has decreased. ")

