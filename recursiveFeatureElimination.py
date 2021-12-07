import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn import tree
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv("/Users/andrewsayadi/Downloads/US_Accidents_Dec20_updated.csv")
X = df.head(1000)
X = df.drop(['ID',
               'Severity',
               'Start_Lat',
               'End_Lat',
               'Start_Lng',
               'End_Lng',
               'Distance(mi)',
               'Start_Time',
               'End_Time',
               'Description'
               ], axis=1)
y = df["Severity"].head(1000)

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

X_labeled = MultiColumnLabelEncoder(columns = ['Street','Side','City','County','State','Zipcode','Country','Timezone','Airport_Code','Weather_Timestamp','Wind_Direction','Weather_Condition','Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop','Turning_Loop','Sunrise_Sunset','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight']).fit_transform(X)

X_labeled = X_labeled.fillna(0)
X_labeled = X_labeled.head(1000)


#Normal Recursive Feature Elimination
# t = tree.DecisionTreeClassifier()
# svc = SVC(kernel="linear", C=1)
# rfe = RFE(estimator=t, n_features_to_select=10, step=1) #using svc takes a long time
# rfe.fit(X_labeled, y)

# indices = rfe.get_support(indices=True)

# for index in indices:
#     print(X_labeled.columns[index])



#Recursive Feature Elimination with Cross Validation
t = tree.DecisionTreeClassifier()
rfecv = RFECV(
    estimator=t,
    step=1,
    cv=StratifiedKFold(5),
    scoring="accuracy",
    min_features_to_select=10,
)
rfecv.fit(X_labeled, y)

print("Optimal number of features : %d" % rfecv.n_features_)