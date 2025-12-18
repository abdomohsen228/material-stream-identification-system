# src/models/svm.py

import pandas as pd
from pathlib import Path
import joblib
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "data" / "extracted_features.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Load features
# ------------------------------------------------------------------
print("Loading CNN features...")
df = pd.read_csv(DATA_CSV)

if df.empty:
    raise ValueError("extracted_features.csv is empty. Run feature extraction first.")

print("Original class distribution:")
print(df["label"].value_counts())

# ------------------------------------------------------------------
# Upsample all minority classes to match the largest class
# ------------------------------------------------------------------
max_size = df['label'].value_counts().max()
df_list = []

for label in df['label'].unique():
    df_label = df[df['label'] == label]
    if len(df_label) < max_size:
        df_label = resample(
            df_label,
            replace=True,
            n_samples=max_size,
            random_state=42
        )
    df_list.append(df_label)

df = pd.concat(df_list)

print("\nAfter full upsampling:")
print(df["label"].value_counts())

X = df.drop(columns=["label", "image"]).values
y = df["label"].values

print(f"Samples: {X.shape[0]}, Feature dim: {X.shape[1]}")

# ------------------------------------------------------------------
# Train / Test split (stratified, fixed random state)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------------
# Pipeline: Scaler -> PCA -> SVM
# Fixed random_state for stability
# ------------------------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=256, whiten=True, random_state=42)),
    ("svm", SVC(
        kernel="rbf",
        C=10,              # fixed best value from experimentation
        gamma=0.001,       # fixed best value
        probability=True,
        class_weight="balanced",
        random_state=42
    ))
])

# ------------------------------------------------------------------
# Train the model
# ------------------------------------------------------------------
print("\nTraining SVM pipeline with fixed parameters...")
pipeline.fit(X_train, y_train)

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nFINAL ACCURACY: {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------------
# Save full pipeline
# ------------------------------------------------------------------
model_path = MODEL_DIR / "svm_resnet50_pipeline.pkl"
joblib.dump(pipeline, model_path)

print(f"Saved model pipeline to: {model_path}")
