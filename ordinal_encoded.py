import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import StratifiedKFold

# this is the entire dataset undersampled, so you can drop/select whatever
data = pd.read_csv("/Users/andrewsayadi/Downloads/random_undersampled_full.csv")

# play around with this based on the feature selection results I sent
X = data[
    [
        "City",
        "Crossing",
        "Pressure(in)",
        "State",
        "Timezone",
        "Traffic_Signal",
        "Wind_Chill(F)",
        "Zipcode",
        "Astronomical_Twilight",
    ]
]

y = data["Severity"]

# grab float and string columns
float_cols = X.select_dtypes(include=["float64"]).columns
str_cols = X.select_dtypes(include=["object"]).columns

# handle NaNs
X.loc[:, float_cols] = X.loc[:, float_cols].fillna(0)
X.loc[:, str_cols] = X.loc[:, str_cols].fillna("")

# encode string features
ord_enc = OrdinalEncoder()

for feature in str_cols:
    X[feature] = ord_enc.fit_transform(X[[feature]])

# check types
#print(X.dtypes)

t = tree.DecisionTreeClassifier()

#add as many models as you want here
count = 1
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    print("ITERATION: ", count)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    lr = tree.DecisionTreeClassifier()
    lr.fit(X_train, y_train)
    print("decision tree: ", lr.score(X_test, y_test))
    plot_confusion_matrix(lr, X_test, y_test, cmap='Blues').ax_.set(title='Decision Tree') 
    plt.show()

    rf = RandomForestClassifier(n_estimators=40)
    rf.fit(X_train, y_train)
    print("Random Forest: ", rf.score(X_test, y_test))
    plot_confusion_matrix(rf, X_test, y_test, cmap='Blues').ax_.set(title='Random Forest')  
    plt.show()

    bagging = BaggingClassifier(base_estimator=t,n_estimators=10, random_state=0).fit(X, y) #score: 97%
    print('Bagging: ', bagging.score(X_test, y_test))
    plot_confusion_matrix(bagging, X_test, y_test)  
    plt.show()

    ada = AdaBoostClassifier(n_estimators=100, random_state=0) #score: 42%
    ada.fit(X, y)
    print('AdaBoost: ', ada.score(X_test, y_test))
    plot_confusion_matrix(ada, X_test, y_test)  
    plt.show()

    logreg = LogisticRegression(random_state=0).fit(X, y) #stops and reaches iteration limit, but outputs 30%
    print('Logistic Regression: ', logreg.score(X_test, y_test))
    plot_confusion_matrix(logreg, X_test, y_test)  
    plt.show()

    neigh = KNeighborsClassifier(n_neighbors=3) #score: 71%, 3 neighbors is best cause less and more decrease accuracy.
    neigh.fit(X, y)
    print('KNeighbors: ', neigh.score(X_test, y_test))
    plot_confusion_matrix(neigh, X_test, y_test)  
    plt.show()

    radius = RadiusNeighborsClassifier()  #score: 99%
    radius.fit(X, y)
    print('Radius Neighbors: ', radius.score(X_test, y_test))
    plot_confusion_matrix(radius, X_test, y_test)  
    plt.show()

    gbc = GradientBoostingClassifier().fit(X, y) # score: 56%
    print('Gradient Boosting: ', gbc.score(X_test, y_test))
    plot_confusion_matrix(gbc, X_test, y_test, cmap='Blues').ax_.set(title='Gradient Boost')
    plt.show()

    xgb_model = XGBClassifier()         #score: 99% at depth = 15 likely over fitting
    xgb_model.fit(X, y)
    print('XGBoost: ', xgb_model.score(X_test, y_test))
    plot_confusion_matrix(xgb_model, X_test, y_test, cmap='Blues').ax_.set(title='XGBoost')
    plt.show()

    count = count + 1