import pandas as pd
from pathlib import Path
import joblib
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "data" / "extracted_features.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

df = pd.read_csv(DATA_CSV)
if df.empty:
    raise ValueError("Feature file is empty. Run feature extraction first.")
max_samples = df["label"].value_counts().max()
balanced_dfs = []
for label in df["label"].unique():
    class_df = df[df["label"] == label]
    if len(class_df) < max_samples:
        class_df = resample(
            class_df,
            replace=True,
            n_samples=max_samples,
            random_state=RANDOM_STATE
        )
    balanced_dfs.append(class_df)

df = pd.concat(balanced_dfs, ignore_index=True)
X = df.drop(columns=["label", "image"]).values
y = df["label"].values
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(whiten=True, random_state=RANDOM_STATE)),
    ("knn", KNeighborsClassifier(n_jobs=-1))
])

param_grid = {
    "pca__n_components": [256, 350, 450],
    "knn__n_neighbors": [3, 5, 7, 11],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "cosine"]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
    n_jobs=-1,
    verbose=1
)

print("Training k-NN model...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Best CV accuracy: {grid_search.best_score_ * 100:.2f}%")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

model_path = MODEL_DIR / "knn_resnet50_pipeline.pkl"
joblib.dump(best_model, model_path)

print(f"\nModel saved to: {model_path}")
