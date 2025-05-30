#1.Choose a binary classification dataset
import pandas as pd

data = pd.read_csv(r"C:\Users\Kamini Shewale\OneDrive\Desktop\AI-ML(internship)\task4\data.csv")
data = data.drop(['Unnamed: 32', 'id'], axis=1)
data['target'] = data['diagnosis'].map({'B': 0, 'M': 1})
data = data.drop(['diagnosis'], axis=1)

print("First 5 rows:")
print(data.head())

print("\nShape of dataset:", data.shape)
print("\nTarget classes:\n", data['target'].value_counts())

