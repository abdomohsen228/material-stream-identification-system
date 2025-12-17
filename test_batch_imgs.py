# test_batch_imgs.py
import sys
import numpy as np
import joblib
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from feature_extraction.cnn_feature_extractor import extract_cnn_features
from models.predict_with_rejection import predict_batch_with_rejection


MODEL_PATH = Path('models/svm_cnn_pipeline.pkl')
pipeline = joblib.load(MODEL_PATH)

BATCH_DIR = Path('data/test_batch')
img_paths = [f for f in BATCH_DIR.glob('*') if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]

CONFIDENCE_THRESHOLD = 0.5  # Adjust based on requirements (0.0-1.0)

X_test, img_names = [], []

for img_path in img_paths:
    try:
        features = extract_cnn_features(img_path)
        if features is not None and len(features) > 0:
            X_test.append(features)
            img_names.append(img_path.name)
    except Exception as e:
        print("Error:", img_path, e)

if len(X_test) == 0:
    raise ValueError("No valid features found for batch images!")

print(f"Predicting {len(X_test)} images with rejection mechanism...")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
print("(Low confidence predictions will be rejected as 'unknown')\n")

results = predict_batch_with_rejection(
    pipeline, X_test,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    unknown_class='unknown'
)

for name, (pred_class, confidence, was_rejected) in zip(img_names, results):
    status = "[REJECTED]" if was_rejected else "[ACCEPTED]"
    print(f"{name:30s} -> {pred_class:10s} ({confidence*100:.2f}%) {status}")
