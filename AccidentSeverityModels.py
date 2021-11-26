from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


data = pd.read_csv("C:/Users/shard/OneDrive/Documents/MSAI/Data Mining/US_Accidents_Dec20_Updated.csv")

#X = data.drop("Severity",axis=1)   #Feature Matrix
X = data[["Start_Lat", "End_Lat"]]           #Using these before we figure out which features to use/one hot encoding possibly neccesary
Y = data["Severity"]          #Target Variable

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))

#Random Forest
rf = RandomForestClassifier(n_estimators = 40)
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))

#K Nearest Neighbors
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
print(neigh.score(X_test, y_test))

#Bayesian Ridge 
reg = linear_model.BayesianRidge()
reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))

#Bernoulli Naive Bayes
bernb = BernoulliNB()
bernb.fit(X_train, y_train)
print(bernb.score(X_test, y_test))

#Multinomail Naive Bayes
mulnb = MultinomialNB()
mulnb.fit(X_train, y_train)
print(neigh.score(X_test, y_test))

#Gaussian Naive Bayes
gaunb = GaussianNB()
gaunb.fit(X_train, y_train)
print(gaunb.score(X_test, y_test))

#Support Vector Regression
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))
