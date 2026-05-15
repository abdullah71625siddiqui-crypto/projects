import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Housing.csv")
df.head()

# Data preprocessing/cleaning
df.isnull().sum()
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Training model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)


# Sample test Data
test_data = pd.DataFrame({
    "Area": [1000, 1500, 2000],
    "Bedrooms": [2, 3, 4],
    "Bathrooms": [1, 2, 3]
})

test_data = test_data.reindex(columns=X.columns, fill_value=0)
predictions = model.predict(test_data)

results = test_data.copy()
results["Predicted Price"] = predictions
print(results)