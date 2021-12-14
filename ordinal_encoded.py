import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

# this is the entire dataset undersampled, so you can drop/select whatever
data = pd.read_csv("./random_undersampled_full.csv")

# play around with this based on the feature selection results I sent
X = data[
    [
        "City",
        "County",
        "State",
        "Zipcode",
        "Airport_Code",
        "Humidity(%)",
        "Weather_Condition",
        "Crossing",
        "Wind_Chill(F)",
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
print(X.dtypes)


# add as many models as you want here
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    lr = tree.DecisionTreeClassifier()
    lr.fit(X_train, y_train)
    print("decision tree: ", lr.score(X_test, y_test))

    rf = RandomForestClassifier(n_estimators=40)
    rf.fit(X_train, y_train)
    print("Random Forest: ", rf.score(X_test, y_test))

