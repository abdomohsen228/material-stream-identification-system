import pandas as pd
from pathlib import Path
import joblib
import numpy as np

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
    ("pca", PCA(n_components=256, whiten=True, random_state=RANDOM_STATE)),
    ("svm", SVC(
        kernel="rbf",
        C=10,
        gamma=0.001,
        probability=True,
        random_state=RANDOM_STATE
    ))
])


print("Training SVM (upsampling only)...")
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {acc * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

model_path = MODEL_DIR / "svm_resnet50_pipeline.pkl"
joblib.dump(pipeline, model_path)
print(f"\nModel saved to: {model_path}")
