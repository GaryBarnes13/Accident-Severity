# Do not run this yet
# We will need to complete feature selection first
# We should also highly consider using one-hot encoding for the features we selct
# For now, I will make sure this runs with label encoding
# The output will be in the form of csv's, which we can split and feed to our models
# We do NOT save csv's to github (too large), so you will need to save the output on your computer

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC

from MultiColumnLabelEncoder import MultiColumnLabelEncoder

data = pd.read_csv("./accidents.zip")

# drop irrelevant columns
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

# get our data ready to balance
# possible GPU annotation here
def preprocess(X, y):
    # subset of data - DELETE BEFORE PUSH
    X = X.head(100)
    y = y.head(100)

    # encode nominal/categorical attributes with label encoding
    X = MultiColumnLabelEncoder(
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

    # fill NaN's with 0
    X = X.fillna(0)

    return X, y


X, y = preprocess(X, y)

# balance with random undersampling
# possible GPU annotation here
def random_undersample(X, y):
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    X_res["Severity"] = y_res
    X_res.to_csv("./gpu/balanced_data/random_undersampled.csv", index=False)


random_undersample(X, y)

# balance with random undersampling
# possible GPU annotation here
def random_oversample(X, y):
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    X_res["Severity"] = y_res
    X_res.to_csv("./gpu/balanced_data/random_oversampled.csv", index=False)


random_oversample(X, y)

# balance witih smote
# used smaller subset of features for simplicity, will need to change
# annotate with GPU here
X = data[
    [
        "County",
        "State",
        "Timezone",
        "Airport_Code",
        "Bump",
        "Give_Way",
        "Roundabout",
        "Stop",
        "Traffic_Signal",
        "Turning_Loop",
    ]
]
y = data["Severity"]

X = X.head(100)
y = y.head(100)


def smotenc_oversample(X, y):
    X = MultiColumnLabelEncoder(
        columns=["County", "State", "Timezone", "Airport_Code"]
    ).fit_transform(X)
    sm = SMOTENC(random_state=42, categorical_features=[4, 5, 6, 7, 8, 9])
    X_res, y_res = sm.fit_resample(X, y)
    X_res["Severity"] = y_res
    X_res.to_csv("./gpu/balanced_data/smotenc_oversampled.csv", index=False)


smotenc_oversample(X, y)


# data.to_csv("./gpu/balanced_data/out.csv", index=False)

