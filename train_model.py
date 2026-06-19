import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("dataset/stroke.csv")

print("Columns in Dataset:")
print(df.columns)
print("\n")

# -----------------------------
# CLEAN COLUMN NAMES
# (Remove spaces if any)
# -----------------------------
df.columns = df.columns.str.strip()

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
df.fillna(df.mean(numeric_only=True), inplace=True)

# -----------------------------
# ENCODE ALL CATEGORICAL COLUMNS AUTOMATICALLY
# -----------------------------
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# -----------------------------
# CHECK TARGET COLUMN NAME
# -----------------------------
if "Stroke" in df.columns:
    target_column = "Stroke"
elif "stroke" in df.columns:
    target_column = "stroke"
else:
    raise Exception("Target column not found! Check column name.")

# -----------------------------
# SPLIT FEATURES & TARGET
# -----------------------------
X = df.drop(target_column, axis=1)
y = df[target_column]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "models/stroke_model.pkl")

print("\nModel Saved Successfully!")
