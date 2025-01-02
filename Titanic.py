import pandas as pd

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Inspecting data
print("Missing values in train data:",("\n"),train_data.isnull().sum())
print("Missing values in test data:", ("\n"),test_data.isnull().sum())
print("Train Data Description",("\n"),train_data.describe())
print("Test Data Description",("\n"),test_data.describe())

