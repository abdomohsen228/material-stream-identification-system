import pandas as pd
from pathlib import Path
import joblib
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "data" / "extracted_features.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Loading CNN features...")
df = pd.read_csv(DATA_CSV)

if df.empty:
    raise ValueError("extracted_features.csv is empty. Run feature extraction first.")

print("Original class distribution:")
print(df["label"].value_counts())

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(whiten=True, random_state=42)),
    ("knn", KNeighborsClassifier(n_jobs=-1))
])
#here we will test multiple combinations, and come up with the best one to go through
param_grid = {
    'pca__n_components': [256, 350, 450], 
    'knn__n_neighbors': [3, 5, 7, 11], 
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'cosine']
}

print("\n" + "="*70)
print("FOCUSED HYPERPARAMETER SEARCH")
print("="*70)
print(f"Total combinations to test: 3 × 4 × 2 × 2 = 48")
print("Estimated time: 2-3 minutes")
print("="*70 + "\n")

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),  # Reduced to 3-fold
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("\n" + "="*70)
print("BEST PARAMETERS FOUND")
print("="*70)
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest CV Accuracy: {grid_search.best_score_ * 100:.2f}%")

best_pipeline = grid_search.best_estimator_

y_pred = best_pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*70)
print("FINAL TEST ACCURACY: " + f"{acc * 100:.2f}%")
print("="*70)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

model_path = MODEL_DIR / "knn_resnet50_pipeline.pkl"
joblib.dump(best_pipeline, model_path)

print("\n" + "="*70)
print(f"Saved model pipeline to: {model_path}")
print("="*70)
