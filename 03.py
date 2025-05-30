#3.Fit a Logistic Regression model.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import pandas as pd

data = pd.read_csv(r"C:\Users\Kamini Shewale\OneDrive\Desktop\AI-ML(internship)\task4\data.csv")
data = data.drop(columns=['Unnamed: 32'])
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
