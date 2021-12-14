import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier


df = pd.read_csv("/Users/andrewsayadi/Downloads/random_undersampled.csv")
X_labeled = df.drop(['Severity'], axis=1)
y = df["Severity"]

#X_train, X_test, y_train, y_test = train_test_split(X_labeled, y, test_size=0.3)

t = tree.DecisionTreeClassifier()

kf = KFold(n_splits = 10)                                     
for train_index, test_index in kf.split(X_labeled):
    X_train, X_test = X_labeled.loc[train_index], X_labeled.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

lr = tree.DecisionTreeClassifier()                     #score: 44%
lr.fit(X_train, y_train)
print('decision tree: ', lr.score(X_test, y_test))
plot_confusion_matrix(lr, X_test, y_test)  
plt.show()

rf = RandomForestClassifier(n_estimators = 40)      #score: 45%
rf.fit(X_train, y_train)
print('Random Forest: ', rf.score(X_test, y_test))
plot_confusion_matrix(rf, X_test, y_test)  
plt.show()

bagging = BaggingClassifier(base_estimator=t,n_estimators=10, random_state=0).fit(X_labeled, y) #score: 97%
print('Bagging: ', bagging.score(X_test, y_test))
plot_confusion_matrix(bagging, X_test, y_test)  
plt.show()

ada = AdaBoostClassifier(n_estimators=100, random_state=0) #score: 42%
ada.fit(X_labeled, y)
print('AdaBoost: ', ada.score(X_test, y_test))
plot_confusion_matrix(ada, X_test, y_test)  
plt.show()

logreg = LogisticRegression(random_state=0).fit(X_labeled, y) #stops and reaches iteration limit, but outputs 30%
print('Logistic Regression: ', logreg.score(X_test, y_test))
plot_confusion_matrix(logreg, X_test, y_test)  
plt.show()

neigh = KNeighborsClassifier(n_neighbors=3) #score: 71%, 3 neighbors is best cause less and more decrease accuracy.
neigh.fit(X_labeled, y)
print('KNeighbors: ', neigh.score(X_test, y_test))
plot_confusion_matrix(neigh, X_test, y_test)  
plt.show()

radius = RadiusNeighborsClassifier()  #score: 99%
radius.fit(X_labeled, y)
print('Radius Neighbors: ', radius.score(X_test, y_test))
plot_confusion_matrix(radius, X_test, y_test)  
plt.show()

gbc = GradientBoostingClassifier().fit(X_labeled, y) # score: 56%
print('Gradient Boosting: ', gbc.score(X_test, y_test))
plot_confusion_matrix(gbc, X_test, y_test)  
plt.show()

xgb_model = XGBClassifier(max_depth = 15)         #score: 99% at depth = 15 likely over fitting
xgb_model.fit(X_labeled, y)
print('XGBoost: ', xgb_model.score(X_test, y_test))
plot_confusion_matrix(xgb_model, X_test, y_test)  
plt.show()