from typing import TYPE_CHECKING, type_check_only
from numpy import testing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

def split():
        df = pd.read_csv("/Users/andrewsayadi/Library/Mobile Documents/com~apple~CloudDocs/Data Mining/Accident-Severity/Accident-Severity/US_Accidents_Dec20_updated.csv")
        
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3) #need to know the data and target we are using
        
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        lr.score(X_test, y_test) #wont run yet cause we havent define data and target

        rf = RandomForestClassifier(n_estimators = 40)
        rf.fit(X_train, y_train)
        rf.score(X_test, y_test) #need to define data and target

        kf = KFold(n_splits = 10)
        for train_index, test_index in kf.split(data): #data is not defined yet
            pass

        folds = StratifiedKFold(n_splits = 10)
        
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

