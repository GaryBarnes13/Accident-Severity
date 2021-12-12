# Do not run this yet
# We will need to complete feature selection first
# We should also highly consider using one-hot encoding for the features we selct
# For now, I will make sure this runs with label encoding
# The output will be in the form of csv's, which we can split and feed to our models
# We do NOT save csv's to github (too large), so you will need to save the output on your computer

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks

from MultiColumnLabelEncoder import MultiColumnLabelEncoder

data = pd.read_csv("./accidents.zip")

# drop irrelevant columns
X = data[
    [
        "Airport_Code",
        "Amenity",
        "Astronomical_Twilight",
        "Bump",
        "City",
        "Civil_Twilight",
        "County",
        "Crossing",
        "Give_Way",
        "Humidity(%)",
        "Junction",
        "Nautical_Twilight",
        "No_Exit",
        "Precipitation(in)",
        "Pressure(in)",
        "Railway",
        "State",
        "Station",
        "Stop",
        "Sunrise_Sunset",
        "Temperature(F)",
        "Timezone",
        "Traffic_Signal",
        "Visibility(mi)",
        "Weather_Condition",
        "Wind_Chill(F)",
        "Wind_Direction",
        "Wind_Speed(mph)",
        "Zipcode",
    ]
]

# set target as severity
y = data["Severity"]

# get our data ready to balance
# possible GPU annotation here
def preprocess(X, y):

    # encode nominal/categorical attributes with label encoding
    X = MultiColumnLabelEncoder(
        columns=[
            "Airport_Code",
            "Amenity",
            "Astronomical_Twilight",
            "Bump",
            "City",
            "Civil_Twilight",
            "County",
            "Crossing",
            "Give_Way",
            "Junction",
            "Nautical_Twilight",
            "No_Exit",
            "Railway",
            "State",
            "Station",
            "Stop",
            "Sunrise_Sunset",
            "Timezone",
            "Traffic_Signal",
            "Weather_Condition",
            "Wind_Direction",
            "Zipcode",
        ]
    ).fit_transform(X)

    # Change Temp
    X["Temperature(Kel)"] = 273.5 + ((X["Temperature(F)"] - 32.0) * (5.0 / 9.0))
    X["Wind_Chill(Kel)"] = 273.5 + ((X["Wind_Chill(F)"] - 32.0) * (5.0 / 9.0))

    # Dropping F values to get rid of negatives
    X = X.drop(["Temperature(F)", "Wind_Chill(F)"], axis=1)

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


def tomek_links(X, y):
    tl = TomekLinks()
    X_res, y_res = tl.fit_resample(X, y)
    X_res["Severity"] = y_res
    X_res.to_csv("./gpu/balanced_data/tomek_links.csv", index=False)


tomek_links(X, y)
