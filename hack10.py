import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from pandas import Series
from numpy.random import randn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from keras.utils import np_utils
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import Imputer 
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.wrappers import TimeDistributed
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from sklearn.grid_search import GridSearchCV

#dictionary
grades = {0:"Grade 1",
          1:"Grade 2",
          2:"Grade 3",
          3:"Grade 4",
          4:"Grade 5" 
          }


#importing csv files
train=pd.read_csv("train.csv")
building_structure=pd.read_csv("Building_Structure.csv")
building_ownership=pd.read_csv("Building_Ownership_Use.csv")

col_merge = ['has_geotechnical_risk','has_geotechnical_risk_fault_crack','has_geotechnical_risk_flood',
             'has_geotechnical_risk_land_settlement','has_geotechnical_risk_landslide','has_geotechnical_risk_liquefaction',
             'has_geotechnical_risk_other','has_geotechnical_risk_rock_fall']

col_merge1 = ['has_secondary_use','has_secondary_use_agriculture','has_secondary_use_hotel',
              'has_secondary_use_rental','has_secondary_use_institution','has_secondary_use_school',
              'has_secondary_use_industry','has_secondary_use_health_post','has_secondary_use_gov_office',
              'has_secondary_use_use_police','has_secondary_use_other']

building_ownership['usage'] = (building_ownership[col_merge1].sum(axis=1))/11
building_ownership.drop(col_merge1, axis=1, inplace=True)


#dropping some columns which are not in use
#train.drop(['vdcmun_id'], axis=1, inplace=True)
building_structure.drop(['district_id','vdcmun_id'], axis=1, inplace=True)
building_ownership.drop(['district_id','vdcmun_id','ward_id'], axis=1, inplace=True)



#merging attributes in train and test files from different files
result = pd.merge(building_structure, building_ownership, on='building_id')
res_train=pd.merge(train,result,on="building_id")

#filling all the null values in the column with mean values in the column
res_train = res_train.fillna(method="ffill")

res_train['risk'] = (res_train[col_merge].sum(axis=1))/8
res_train.drop(col_merge, axis=1, inplace=True)

#splitting coulmns into X and Y training datasets
X_train_org=res_train.loc[:, res_train.columns != 'damage_grade']
Y_train_org=res_train.loc[:, ['damage_grade']]

#creating training and validation data from training data
X_train,X_valid,Y_train,Y_valid=train_test_split(X_train_org,Y_train_org,test_size=0.3)

#storing building id in a seperate variable data frame
build_id_dummy = X_valid.loc[:, ['building_id']]
X_train = X_train.loc[:, X_train.columns != 'building_id']
X_valid = X_valid.loc[:, X_valid.columns != 'building_id']


#converting categorical data to numerical data using Label Encoder
X_train = pd.get_dummies(X_train, drop_first=True)
X_valid = pd.get_dummies(X_valid, drop_first=True)

#encoding output variables
Y_train, uniques1 = pd.factorize(Y_train['damage_grade'], sort=True)
Y_valid, uniques2 = pd.factorize(Y_valid['damage_grade'], sort=True)
#Y_train = pd.get_dummies(Y_train)
#Y_valid = pd.get_dummies(Y_valid)

#selecting  predictors as all headers in the train csv
predictors=list(X_train)

'''#plotting the graph to see the best arguments or predictors which give best result
plt.bar(np.arange(len(predictors)), rf.feature_importances_)
plt.xticks(np.arange(len(predictors)), predictors, rotation='vertical')
plt.show()'''

#adding Random forest classifier model
rf=RandomForestClassifier(n_estimators = 100)
rf.fit(X_train, Y_train)


## Predicting the values
train_predict = rf.predict(X_train)
valid_predict = rf.predict(X_valid)

## Checking the accuracy
print("train accuracy::",accuracy_score(Y_train, train_predict))
print("test accuracy::",accuracy_score(Y_valid, valid_predict))



#extracting testing data from csv files
#importing csv files
test=pd.read_csv("test.csv")
building_structure1=pd.read_csv("Building_Structure.csv")
building_ownership1=pd.read_csv("Building_Ownership_Use.csv")

building_ownership1['usage'] = (building_ownership1[col_merge1].sum(axis=1))/11
building_ownership1.drop(col_merge1, axis=1, inplace=True)

#dropping some columns which are not in use
#test.drop(['vdcmun_id'], axis=1, inplace=True)
building_structure1.drop(['district_id','vdcmun_id'], axis=1, inplace=True)
building_ownership1.drop(['district_id','vdcmun_id','ward_id'], axis=1, inplace=True)

#merging attributes in train and test files from different files
test_result = pd.merge(building_structure1, building_ownership1, on='building_id')
res_test=pd.merge(test,test_result,on="building_id")

res_test['risk'] = (res_test[col_merge].sum(axis=1))/8
res_test.drop(col_merge, axis=1, inplace=True)

#storing building id in a seperate variable data frame
build_id = res_test.loc[:, ['building_id']]
X_test = res_test.loc[:, res_test.columns != 'building_id']

#encoding testing data
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.fillna(method="ffill")
#predicting test dataset


train_prediction = rf.predict(X_test)

arr1=[]
for i in range(0,len(train_prediction)):
    arr1.append(grades[train_prediction[i]])

submission = pd.DataFrame({"building_id": build_id["building_id"], "damage_grade": arr1})
submission.to_csv("submission9.csv", index=False)


    
    
