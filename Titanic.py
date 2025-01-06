import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Inspecting data
#print("Missing values in train data:",("\n"),train_data.isnull().sum())
#print("Missing values in test data:", ("\n"),test_data.isnull().sum())
#print("Train Data Description",("\n"),train_data.describe())
#print("Test Data Description",("\n"),test_data.describe())

# Data Cleaning
train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())
test_data["Age"] = test_data["Age"].fillna(test_data["Age"].median())
train_data.drop("Cabin", axis=1, inplace=True)
test_data.drop("Cabin", axis=1, inplace=True)
train_data["Embarked"] = train_data["Embarked"].fillna(train_data["Embarked"].mode()[0])

print(train_data.isnull().sum())

# Creating new features
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
train_data['Embarked'] = train_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
test_data['Embarked'] = test_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

#Showing data after cleaning
sns.barplot(x='Sex', y='Survived', data=train_data, palette=["pink", "blue"])
plt.title("Survival Rate by Sex")
plt.xticks(ticks= [0,1], labels= ["Females", "Males"])
plt.show()
