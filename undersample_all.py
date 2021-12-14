import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

data = pd.read_csv("./accidents.zip")

X = data.drop(["Severity"], axis=1)
y = data["Severity"]

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)
X_res["Severity"] = y_res
X_res.to_csv("./random_undersampled_full.csv", index=False)
