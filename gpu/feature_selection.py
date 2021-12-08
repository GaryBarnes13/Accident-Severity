import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from datetime import datetime
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, chi2

from MultiColumnLabelEncoder import MultiColumnLabelEncoder

# read in accident data
data = pd.read_csv("./accidents.zip")

f = open("./gpu/feature_selection_results.txt", "a")

f.write("--- RUN BEGIN ---\n")
f.write(f"{datetime.now()}\n\n")

# drop the attributes we know we should not use to predict severity
X = data.drop(
    [
        "ID",
        "Severity",
        "Start_Lat",
        "End_Lat",
        "Start_Lng",
        "End_Lng",
        "Distance(mi)",
        "Start_Time",
        "End_Time",
        "Description",
        "Number",
        "Side",
        "Street",
    ],
    axis=1,
)

# set target as severity
y = data["Severity"]

# encode nominal/categorical attributes with label encoding
X_labeled = MultiColumnLabelEncoder(
    columns=[
        "City",
        "County",
        "State",
        "Zipcode",
        "Country",
        "Timezone",
        "Airport_Code",
        "Weather_Timestamp",
        "Wind_Direction",
        "Weather_Condition",
        "Amenity",
        "Bump",
        "Crossing",
        "Give_Way",
        "Junction",
        "No_Exit",
        "Railway",
        "Roundabout",
        "Station",
        "Stop",
        "Traffic_Calming",
        "Traffic_Signal",
        "Turning_Loop",
        "Sunrise_Sunset",
        "Civil_Twilight",
        "Nautical_Twilight",
        "Astronomical_Twilight",
    ]
).fit_transform(X)

# convert temp to Kelvin
# Converting F to Kelvin because K Best doesn't accept negative numbers
X_labeled["Temperature(Kel)"] = 273.5 + (
    (X_labeled["Temperature(F)"] - 32.0) * (5.0 / 9.0)
)
X_labeled["Wind_Chill(Kel)"] = 273.5 + (
    (X_labeled["Wind_Chill(F)"] - 32.0) * (5.0 / 9.0)
)

# Dropping F values to get rid of negatives
X_labeled = X_labeled.drop(["Temperature(F)", "Wind_Chill(F)"], axis=1)

# fill NaN's with 0
X_labeled = X_labeled.fillna(0)

X_labeled = X_labeled.head(100)
y = y.head(100)

# create classifier for use below
clf = DecisionTreeClassifier(random_state=0)

# find optimum number of features
rfecv = RFECV(
    estimator=clf, step=1, cv=StratifiedKFold(10), scoring="accuracy",
)
rfecv.fit(X_labeled, y)

# grab optimum number of features and write to results
number_features = rfecv.n_features_
f.write(f"optimum number of features based on RFECV: {number_features}\n\n")

# grab most important features based on SFS
sfs = SequentialFeatureSelector(clf, n_features_to_select=number_features)
sfs.fit(X_labeled, y)

indices = sfs.get_support(indices=True)

# record results of SFS
f.write("SFS Results\n")
for index in indices:
    f.write(f"{X_labeled.columns[index]}, ")

f.write("\n\n")

# Normal Recursive Feature Elimination
rfe = RFE(estimator=clf, n_features_to_select=number_features, step=1)
rfe.fit(X_labeled, y)

indices = rfe.get_support(indices=True)

f.write("RFE Results\n")
for index in indices:
    f.write(f"{X_labeled.columns[index]}, ")

f.write("\n\n")

f.write("Select K-Best Results\n\n")

skb = SelectKBest(chi2, k=10).fit(X_labeled, y)

indices = skb.get_support(indices=True)

for index in indices:
    f.write(f"{X_labeled.columns[index]}, ")


f.write("\n\n--- RUN END ---\n\n\n\n")
f.close()
