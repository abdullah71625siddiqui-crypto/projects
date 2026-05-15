import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("performance.csv")
print(df.isna().sum())

# Drop useless columns
df = df.drop("Age", axis=1)
df = df.drop("Student_ID", axis=1)  # ✅ FIX: drop ID, it's not a feature

# Drop duplicates
df = df.drop_duplicates()

# Fill missing values
df["Gender"] = df["Gender"].fillna("Unknown")
df["Assignment_Score"] = df["Assignment_Score"].fillna(df["Assignment_Score"].median())
df["Internal_Marks"] = df["Internal_Marks"].fillna(df["Internal_Marks"].median())
df["Internet_Access"] = df["Internet_Access"].fillna("Unknown")
df["Previous_GPA"] = df["Previous_GPA"].fillna(df["Previous_GPA"].mode()[0])
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].mode()[0])
df["Study_Hours"] = df["Study_Hours"].fillna(df["Study_Hours"].mode()[0])

print(df.shape)

# Encode target label
le = LabelEncoder()
df["Final_Result"] = le.fit_transform(df["Final_Result"])
print("Label classes:", le.classes_)  # ✅ shows what 0, 1, 2 map to

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=["Gender", "Participation", "Internet_Access", "Family_Background"])

# ✅ Convert bool columns to int (fixes sklearn warnings)
df_encoded = df_encoded.astype({col: int for col in df_encoded.select_dtypes(include='bool').columns})

df_encoded.to_csv("encoded_student_data.csv", index=False)
print(df_encoded.head())

# Split features and target
X = df_encoded.drop("Final_Result", axis=1)
y = df_encoded["Final_Result"]

# ✅ Save columns AFTER dropping Final_Result and Student_ID
pickle.dump(list(X.columns), open("columns.pkl", "wb"))
print("Saved columns:", list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

y_dummy = dummy.predict(X_test)
y_lr = lr.predict(X_test)
y_dt = dt.predict(X_test)

print("\nModel Comparison:")
print("Dummy:", accuracy_score(y_test, y_dummy))
print("Logistic Regression:", accuracy_score(y_test, y_lr))
print("Decision Tree:", accuracy_score(y_test, y_dt))

models = {
    "Dummy": (dummy, accuracy_score(y_test, y_dummy)),
    "Logistic Regression": (lr, accuracy_score(y_test, y_lr)),
    "Decision Tree": (dt, accuracy_score(y_test, y_dt))
}

best_name = max(models, key=lambda k: models[k][1])
best_model_obj = models[best_name][0]
print("Best Model:", best_name)
print("Best Accuracy:", models[best_name][1])

# ✅ Save the best model
pickle.dump(best_model_obj, open("model.pkl", "wb"))
print("model.pkl saved!")

import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("performance.csv")

print("Unique values in Final_Result:", df["Final_Result"].unique())
print("Value counts:\n", df["Final_Result"].value_counts())
print("Nulls in Final_Result:", df["Final_Result"].isna().sum())

import pandas as pd
df = pd.read_csv("performance.csv")
print(df.shape)
