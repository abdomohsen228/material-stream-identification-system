# src/models/knn.py
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import numpy as np

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "data/cnn_features.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load CNN features
print("Loading CNN features...")
df = pd.read_csv(DATA_CSV)
if df.empty:
    raise ValueError("CNN features CSV is empty!")

# Features and labels
X = df.drop(columns=['label', 'image']).values
y = df['label'].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=256, whiten=True)),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
    'knn__p': [1, 2, 3]  # used for minkowski
}

print("Tuning k-NN hyperparameters...")
grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best k-NN params:", grid.best_params_)

y_prob = best_model.predict_proba(X_test)
threshold = 0.6  
y_pred_conf = []
classes = le.classes_
for probs in y_prob:
    max_idx = np.argmax(probs)
    if probs[max_idx] >= threshold:
        y_pred_conf.append(classes[max_idx])
    else:
        y_pred_conf.append("unknown")

y_test_names = le.inverse_transform(y_test)

print("\nFinal Accuracy with threshold:", round(accuracy_score(y_test_names, y_pred_conf) * 100, 2), "%")
print("Classification Report:")
print(classification_report(y_test_names, y_pred_conf, zero_division=0))

joblib.dump(best_model, MODEL_DIR / "knn_cnn_pipeline.pkl")
joblib.dump(le, MODEL_DIR / "label_encoder.pkl")
print("Saved pipeline: knn_cnn_pipeline.pkl")
print("Saved label encoder: label_encoder.pkl")