# src/models/svm.py

import pandas as pd
from pathlib import Path
import joblib
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
# from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

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
    raise ValueError("cnn_features.csv is empty. Run feature extraction first.")
# ---------------------------------------------------------------
# Fix severe class imbalance (trash class)
# ---------------------------------------------------------------
print("Original class distribution:")
print(df["label"].value_counts())

df_majority = df[df.label != "trash"]
df_trash = df[df.label == "trash"]

# Upsample trash to match a medium-sized class (e.g. paper)
target_size = df[df.label == "paper"].shape[0]

df_trash_upsampled = resample(
    df_trash,
    replace=True,
    n_samples=target_size,
    random_state=42
)

df = pd.concat([df_majority, df_trash_upsampled])

print("\nAfter trash upsampling:")
print(df["label"].value_counts())

X = df.drop(columns=["label", "image"]).values
y = df["label"].values

print(f"Samples: {X.shape[0]}, Feature dim: {X.shape[1]}")

# ------------------------------------------------------------------
# Train / Test split
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------------
# Pipeline: Scaler -> SVM
# (NO PCA for CNN features)
# ------------------------------------------------------------------
from sklearn.decomposition import PCA

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=256, whiten=True)),
    ("svm", SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced"
    ))
])


# ------------------------------------------------------------------
# Hyperparameter grid (correct range)
# ------------------------------------------------------------------
param_grid = {
    "svm__C": [0.1, 1, 5, 10, 50, 100],
    "svm__gamma": ["scale", 1e-3, 1e-2, 1e-1]
}

print("Tuning SVM hyperparameters...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest parameters:")
print(grid.best_params_)

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nFINAL ACCURACY: {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------------
# Save full pipeline
# ------------------------------------------------------------------
model_path = MODEL_DIR / "svm_resnet50_pipeline.pkl"
joblib.dump(best_model, model_path)

print(f"Saved model pipeline to: {model_path}")
