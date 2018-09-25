DIR='/Users/ravali/Desktop'
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
df1 = pd.read_csv(DIR+'/sp1.csv', delimiter=',')
#print (df1)
#print(df1.head())
#print(df1.dtypes) 
df1=df1.dropna(how='any')
obj_df = df1.select_dtypes(include=['object']).copy()
obj_df.head()
obj_df["Play"] = obj_df["Play"].astype('category')
obj_df["Play"] = obj_df["Play"].cat.codes
obj_df["ActSceneLine"] = obj_df["ActSceneLine"].astype('category')
obj_df["ActSceneLine"] = obj_df["ActSceneLine"].cat.codes
obj_df["Player"] = obj_df["Player"].astype('category')
obj_df["Player"] = obj_df["Player"].cat.codes
obj_df["PlayerLine"] = obj_df["PlayerLine"].astype('category')
obj_df["PlayerLine"] = obj_df["PlayerLine"].cat.codes
#print(obj_df.head())
df1 = pd.concat([df1, obj_df], axis=1)
df1=df1.drop('Play',1)
df1=df1.drop('ActSceneLine',1)
df1=df1.drop('PlayerLine',1)
df1=df1.drop('Player',1)
df1 = pd.concat([df1, obj_df], axis=1)
#print(df1['Play'])
Y=df1.Player #output label
X=df1.drop('Player',axis=1)#set without outputlabel testset
#print(X)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
#print ("\nX_train:\n")
#print(X_train.head())
#print (X_train.shape)
#print ("\nX_test:\n")
#print(X_test.head())
#print (X_test.shape)
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
redictions = logisticRegr.predict(X_test)
print(score = logisticRegr.score(X_test, y_test))
