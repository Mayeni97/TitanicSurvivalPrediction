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

