import sys
from pathlib import Path
import joblib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from feature_extraction.cnn_feature_extractor import extract_cnn_features
from models.predict_with_rejection import predict_svm_with_rejection
MODEL_DIR = Path("models")
TEST_DIR = Path("test_images")  

CONFIDENCE_THRESHOLD = 0.5  # Adjust based on requirements (0.0-1.0)

svm = joblib.load(MODEL_DIR / "svm_cnn_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
pca = joblib.load(MODEL_DIR / "pca.pkl")

print("Predicting unknown images with rejection mechanism...")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
print("(Low confidence predictions will be rejected as 'unknown')\n")

for img_path in TEST_DIR.iterdir():
    if not img_path.is_file():
        continue
    try:
        features = extract_cnn_features(img_path)
        if features is None:
            print(f"{img_path.name:30s} → {'unknown':10s} (0.00%) [REJECTED - Feature extraction failed]")
            continue
        pred_class, confidence, was_rejected, all_probs = predict_svm_with_rejection(
            svm, scaler, pca, features,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            unknown_class='unknown'
        )
        status = "[REJECTED]" if was_rejected else "[ACCEPTED]"
        print(f"{img_path.name:30s} → {pred_class:10s} ({confidence*100:.2f}%) {status}")
    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")
