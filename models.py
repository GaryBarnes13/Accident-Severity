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

df = pd.read_csv("/Users/andrewsayadi/Downloads/US_Accidents_Dec20_updated.csv")
X = df.head(10000)
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

t = tree.DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y, test_size=0.3)

# lr = tree.DecisionTreeClassifier()
# lr.fit(X_train, y_train)
# print('train test split decision tree: ', lr.score(X_test, y_test))

# rf = RandomForestClassifier(n_estimators = 40)      #score: 73%
# rf.fit(X_train, y_train)
# print('train test split Random Forest: ', rf.score(X_test, y_test))

# bagging = BaggingClassifier(base_estimator=t,n_estimators=10, random_state=0).fit(X_labeled, y) #score: 97%
# print(bagging.score(X_test, y_test))

# ada = AdaBoostClassifier(n_estimators=100, random_state=0) #score: 70%
# ada.fit(X_labeled, y)
# print(ada.score(X_test, y_test))

# logreg = LogisticRegression(random_state=0).fit(X_labeled, y) #stops and reaches iteration limit, but outputs 61%
# print(logreg.score(X_test, y_test))

# neigh = KNeighborsClassifier(n_neighbors=3) #score: 80%, 3 neighbors is best cause less and more decrease accuracy.
# neigh.fit(X_labeled, y)
# print(neigh.score(X_test, y_test))

radius = RadiusNeighborsClassifier(radius=1.6)  #score: 99%
radius.fit(X_labeled, y)
print(radius.score(X_test, y_test))

# gbc = GradientBoostingClassifier().fit(X_labeled, y) # score: 84%
# print(gbc.score(X_test, y_test))

# xgb_model = xgb.XGBRegressor()         #score: 99% not sure if thats actually accurate
# xgb_model.fit(X_labeled, y)
# print(xgb_model.score(X_test, y_test))