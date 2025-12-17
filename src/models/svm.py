# src/models/svm.py
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
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

# Pipeline: Scaler -> PCA -> SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95, random_state=42)),  
    ('svm', SVC(probability=True, class_weight='balanced'))
])

param_grid = {
    'svm__C': [1, 5, 10],
    'svm__gamma': ['scale', 0.01, 0.1],
    'svm__kernel': ['rbf']
}

print(" Tuning SVM hyperparameters...")
grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best SVM params:", grid.best_params_)

# Evaluation
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n FINAL ACCURACY:", round(acc * 100, 2), "%")
print(" Classification Report:")
print(classification_report(y_test, y_pred))

# Save full pipeline
joblib.dump(best_model, MODEL_DIR / "svm_cnn_pipeline.pkl")
print(" Saved pipeline: svm_cnn_pipeline.pkl")
