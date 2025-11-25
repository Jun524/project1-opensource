import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------
df = pd.read_csv("processed_clothing_data.csv")

# Common input features (X)
X = df[["gender", "style", "color", "price"]]

# Targets (y)
y_top = df["top"]
y_bottom = df["bottom"]
y_outer = df["outer"]

# ---------------------------------------------------
# 2. Preprocessing (One-Hot Encoding)
# ---------------------------------------------------
categorical_cols = ["gender", "style", "color", "price"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="drop"
)

# ---------------------------------------------------
# 3. Create Model Builder Function
# ---------------------------------------------------
def build_and_train_model(X, y):
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    model.fit(X, y)
    return model

# ---------------------------------------------------
# 4. Train Models
# ---------------------------------------------------
print("Training top model...")
model_top = build_and_train_model(X, y_top)

print("Training bottom model...")
model_bottom = build_and_train_model(X, y_bottom)

print("Training outer model...")
model_outer = build_and_train_model(X, y_outer)

# ---------------------------------------------------
# 5. Save Models
# ---------------------------------------------------
joblib.dump(model_top, "model_top.pkl")
joblib.dump(model_bottom, "model_bottom.pkl")
joblib.dump(model_outer, "model_outer.pkl")

print("🎉 All models trained and saved successfully!")
print("Saved files: model_top.pkl, model_bottom.pkl, model_outer.pkl")
