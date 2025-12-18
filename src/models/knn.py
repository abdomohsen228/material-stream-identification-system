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

# Encode labels (string -> integer)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Pipeline: Scaler -> PCA -> k-NN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95, random_state=42)),
    ('knn', KNeighborsClassifier())
])

# Hyperparameter grid
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski'],
    'knn__p': [1, 2]
}

# GridSearchCV
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

# Evaluation
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Final Accuracy:", round(acc * 100, 2), "%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# Save full pipeline and label encoder
joblib.dump(best_model, MODEL_DIR / "knn_cnn_pipeline.pkl")
joblib.dump(le, MODEL_DIR / "label_encoder.pkl")
print("Saved pipeline: knn_cnn_pipeline.pkl")
print("Saved label encoder: label_encoder.pkl")