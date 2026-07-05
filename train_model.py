import pandas as pd
import pickle
import os
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -----------------------------
# CREATE MODEL FOLDER SAFELY
# -----------------------------
os.makedirs("model", exist_ok=True)

# -----------------------------
# LOAD DATASET
# -----------------------------
with ZipFile("dataset/archive.zip", 'r') as z:
    with z.open("Human Development Index - Full.csv") as f:
        df = pd.read_csv(f)

# Clean column names
df.columns = df.columns.str.strip()

# -----------------------------
# USE ONLY ONE FEATURE (VERY IMPORTANT)
# -----------------------------
feature_column = "HDI Rank (2021)"
target_column = "Human Development Index (2021)"

# Keep only required columns and remove missing values
df = df[[feature_column, target_column]].dropna()

# Input (X) MUST be 2D
X = df[[feature_column]]

# Output (y)
y = df[target_column]

# -----------------------------
# SPLIT DATA
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# SAVE MODEL
# -----------------------------
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("MODEL TRAINED SUCCESSFULLY ✔")
print("FEATURE USED:", feature_column)