DIR='/Users/ravali/Desktop/DataScience-Classification/data'
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns 
predictors = pd.read_csv(DIR+'/train_predictors.csv', delimiter=',')
test_predictors = pd.read_csv(DIR+'/test_predictors.csv', delimiter=',')
targets = pd.read_csv(DIR+'/train_targets.csv', delimiter=',')
predictors = predictors[pd.to_numeric(predictors.ActSceneLine, errors='coerce').notnull()]
predictors.ActSceneLine = predictors.ActSceneLine.astype(float)
print(predictors.dtypes)
predictors = predictors[pd.to_numeric(predictors.Dataline, errors='coerce').notnull()]
predictors.Dataline = predictors.ActSceneLine.astype(float)
predictors = predictors[pd.to_numeric(predictors.Play, errors='coerce').notnull()]
predictors.Play = predictors.Play.astype(float)
predictors = predictors[pd.to_numeric(predictors.PlayerLine, errors='coerce').notnull()]
predictors.PlayerLine = predictors.PlayerLine.astype(float)
print(predictors.describe())
sns.FacetGrid(predictors, hue="Player").map(plt.scatter, "ActSceneLine", "PlayerLine").add_legend()
lr=LogisticRegression(class_weight= None)
lr.fit(predictors, targets)
test_out = lr.predict(test_predictors)
print (type(test_out)) 
test_out.to_csv('/Users/ravali/Desktop/DataScience-Classification/data/output.csv', sep=',') 