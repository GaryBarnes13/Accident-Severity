import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

#from typing import TYPE_CHECKING, type_check_only
from numpy import testing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn import tree


def split():

        # X_train, X_test, y_train, y_test = train_test_split(X_labeled, y, test_size=0.3) #need to know the data and target we are using
        
        # lr = tree.DecisionTreeClassifier()
        # lr.fit(X_train, y_train)
        # print('train test split decision tree: ', lr.score(X_test, y_test))
    

        # rf = RandomForestClassifier(n_estimators = 40)
        # rf.fit(X_train, y_train)
        # print('train test split Random Forest: ', rf.score(X_test, y_test))




        # kf = KFold(n_splits = 10)                                     
        # for train_index, test_index in kf.split(X_labeled):
        #     #print("TRAIN:", train_index, "TEST:", test_index)
        #     X_train_KFold, X_test_KFold = X_labeled.loc[train_index], X_labeled.loc[test_index]
        #     y_train_KFold, y_test_KFold = y.loc[train_index], y.loc[test_index]

        # lr_KFold = tree.DecisionTreeClassifier()
        # lr_KFold.fit(X_train_KFold, y_train_KFold)
        # print('KFold decision tree: ', lr_KFold.score(X_test_KFold, y_test_KFold))

        # rf_KFold = RandomForestClassifier(n_estimators = 40)
        # rf_KFold.fit(X_train_KFold, y_train_KFold)
        # print('KFold Random Forest: ', rf_KFold.score(X_test_KFold, y_test_KFold))



        folds = StratifiedKFold(n_splits = 10)
        for train_index, test_index in folds.split(X_labeled, y):
             #print("TRAIN:", train_index, "TEST:", test_index)
             X_train_stratified, X_test_stratified = X_labeled.loc[train_index], X_labeled.loc[test_index]
             y_train_stratified, y_test_stratified = y.loc[train_index], y.loc[test_index]

        # lr_stratified = tree.DecisionTreeClassifier()
        # lr_stratified.fit(X_train_stratified, y_train_stratified)
        # print('Stratified KFold decision tree: ', lr_stratified.score(X_test_stratified, y_test_stratified))

        rf_stratified = RandomForestClassifier(n_estimators = 40)
        rf_stratified.fit(X_train_stratified, y_train_stratified)
        print('Stratified KFold Random Forest: ', rf_stratified.score(X_test_stratified, y_test_stratified))

        
split()



# 1. train test split: specify how large the test size is (.30) and it will take 70% of data for train and other 
#    30% for testing. 
    # this is a very basic approach but it doesnt perform too well. if you were giving a kid 100 math questions and split the 
    # data like this. maybe most of the questions that appeared in the training set were about algebra but the test questions
    # are calculus. Its just not very fair and split evenly. Another problem with this approach is that the training and 
    # testing set will not be uniform, meaning that everytime the train_test_split is run, the train set and test set will 
    # be different than the last time it was ran. This will result in varying scores produced by the models each time they 
    # are run.

# 2. K fold validation: say theres 100 math questions. make 5 folds of 20 questions. Then make fold 1 the test set and the rest
#     are for training. Note the score and then do again with fold 2 as test. Keep doing this for all of the folds and
#     then average all of the scores and that is the final score. This method produces much better results because
    # it creates a variety of testing data and does not leave leave out content that could have happened with approach 1

    #using a stratified KFold is better because it adds more uniformity to the division of categories.

