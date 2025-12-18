# test_batch_imgs.py

import sys
from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from feature_extraction.feature_extraction import extract_resnet50_features
from models.predict_with_rejection import predict_batch_with_rejection

MODEL_PATH = Path("models/svm_resnet50_pipeline.pkl")
pipeline = joblib.load(MODEL_PATH)
BATCH_DIR = Path("test_images")
img_paths = [p for p in BATCH_DIR.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]

CONFIDENCE_THRESHOLD = 0.5

X_test = []
img_names = []

for img_path in img_paths:
    try:
        features = extract_resnet50_features(img_path)
        X_test.append(features)
        img_names.append(img_path.name)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

if not X_test:
    raise ValueError("No valid images found for testing.")

print(f"\nPredicting {len(X_test)} images")
print(f"Confidence threshold = {CONFIDENCE_THRESHOLD}\n")

results = predict_batch_with_rejection(
    pipeline,
    np.array(X_test),
    confidence_threshold=CONFIDENCE_THRESHOLD,
    unknown_class="unknown"
)

for name, (pred, conf, rejected) in zip(img_names, results):
    status = "[REJECTED]" if rejected else "[ACCEPTED]"
    print(f"{name:30s} → {pred:10s} ({conf*100:.2f}%) {status}")
