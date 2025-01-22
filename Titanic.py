import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.model_selection import OridnaryEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score,StratifiedGroupKFold,train_test_split,GridSearchCV


train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# print(train_data.head(),train_data.info(),train_data.describe())

print(train_data.groupby(["Pclass"], as_index=False) ["Survived"].mean())
print(train_data.groupby(['SibSp'], as_index=False) ["Survived"].mean())
print(train_data.groupby(['Parch'], as_index=False) ["Survived"].mean())

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] +1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1