# src/models/knn.py
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "data/cnn_features.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Loading CNN features...")
df = pd.read_csv(DATA_CSV)
if df.empty:
    raise ValueError("CNN features CSV is empty!")

X = df.drop(columns=['label', 'image']).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline: Scaler -> PCA -> KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95, random_state=42)),  
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski'],
    'knn__p': [1, 2]  # For minkowski: 1=manhattan, 2=euclidean
}

print("🔍 Tuning KNN hyperparameters...")
grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("✅ Best KNN params:", grid.best_params_)

# Evaluation
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n🎯 FINAL ACCURACY:", round(acc * 100, 2), "%")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save full pipeline
joblib.dump(best_model, MODEL_DIR / "knn_cnn_pipeline.pkl")
print("💾 Saved pipeline: knn_cnn_pipeline.pkl")
