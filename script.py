import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train=pd.read_csv("/Users/User/Desktop/ML0000/python/1/train.csv")

test=pd.read_csv("/Users/User/Desktop/ML0000/python/1/test.csv")

train=train.drop(["id"],1)
test=test.drop(["id"],1)


# id,Gender,Age,Driving_License,Region_Code,Previously_Insured,Vehicle_Age,Vehicle_Damage,Annual_Premium,Policy_Sales_Channel,Vintage,Response
# 1,Male,44,1,28.0,0,> 2 Years,Yes,40454.0,26.0,217,1

#Converting text to numerical data

train.loc[train['Gender'] == 'Male', 'Gender'] = 1
train.loc[train['Gender'] == 'Female', 'Gender'] = 0
test.loc[test['Gender'] == 'Male', 'Gender'] = 1
test.loc[test['Gender'] == 'Female', 'Gender'] = 0

train.loc[train['Vehicle_Age'] == '> 2 Years', 'Vehicle_Age'] = 2
train.loc[train['Vehicle_Age'] == '1-2 Year', 'Vehicle_Age'] = 1
train.loc[train['Vehicle_Age'] == '< 1 Year', 'Vehicle_Age'] = 0
test.loc[test['Vehicle_Age'] == '> 2 Years', 'Vehicle_Age'] = 2
test.loc[test['Vehicle_Age'] == '1-2 Year', 'Vehicle_Age'] = 1
test.loc[test['Vehicle_Age'] == '< 1 Year', 'Vehicle_Age'] = 0

train.loc[train['Vehicle_Damage'] == 'Yes', 'Vehicle_Damage'] = 1
train.loc[train['Vehicle_Damage'] == 'No', 'Vehicle_Damage'] = 0
test.loc[test['Vehicle_Damage'] == 'Yes', 'Vehicle_Damage'] = 1
test.loc[test['Vehicle_Damage'] == 'No', 'Vehicle_Damage'] = 0

# Setting int types
for col in train.columns:
    train[col] = train[col].astype(np.int32)


#Correlation for every feature with target

# for col in train.columns:
#     if col == 'Response':
#         continue
#     print(col, train[col].corr(train['Response']))


#-----Modeling-------
# U-S-->Kmeans
from sklearn.cluster import KMeans


X = train.drop(['Response'],1)
y = train['Response']

# cf=KMeans(n_clusters=2)
# cf.fit(X)

#Testing accuracy

# train["clusters"]=cf.labels_
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
# print(train['clusters'].value_counts())
# print('Kmeans accuracy: ', accuracy_score(train['Response'], train['clusters']))
# print('Kmeans f1_score: ', f1_score(train['Response'], train['clusters']))

#U-S COPOD anomaly detection model
from pyod.models.copod import COPOD


# response = train['Response']
# train = train.drop(['Response'], axis=1)
# cf = COPOD(contamination=0.15)
# cf.fit(train)
# cluster=cf.predict(train)
#Testing accuracy

# train['clusters'] = cluster
# train['Response'] = response
# print(train['clusters'].value_counts())
# print('Kmeans accuracy: ', accuracy_score(train['Response'], train['clusters']))
# print('Kmeans f1_score: ', f1_score(train['Response'], train['clusters']))

#(Supervised)Logistic Regression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression

cf=LogisticRegression()
cf.fit(X_train,y_train)
prd=cf.predict(X_test)
print(accuracy_score(y_test,prd))

#Confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, prd)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g')

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
