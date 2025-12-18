# src/models/test_rejection_mechanism.py

import sys
from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from models.rejection_mechanism import SVMRejectionMechanism, DEFAULT_THRESHOLDS
from models.predict_with_rejection import predict_pipeline_with_rejection
from feature_extraction.feature_extraction import extract_resnet50_features

def test_rejection_mechanisms():
    classes = np.array(['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'])

    test_cases = [
        ("High Confidence", np.array([0.85, 0.05, 0.03, 0.02, 0.03, 0.02])),
        ("Low Confidence", np.array([0.35, 0.20, 0.15, 0.12, 0.10, 0.08])),
        ("Borderline", np.array([0.55, 0.20, 0.10, 0.08, 0.05, 0.02]))
    ]

    rejection = SVMRejectionMechanism(confidence_threshold=0.5)

    print("\nTesting rejection logic only\n")
    for name, probs in test_cases:
        pred, conf, rejected = rejection.predict_with_proba(probs, classes)
        status = "REJECTED" if rejected else "ACCEPTED"
        print(f"{name:15s} → {pred:10s} ({conf:.3f}) [{status}]")


def test_pipeline_on_images():
    pipeline = joblib.load(ROOT / "models" / "svm_resnet50_pipeline.pkl")
    test_dir = ROOT / "test_images"

    print("\nTesting pipeline with rejection on real images\n")

    for img_path in test_dir.glob("*.jpg"):
        features = extract_resnet50_features(img_path).reshape(1, -1)
        pred, conf, rejected = predict_pipeline_with_rejection(
            pipeline, features, confidence_threshold=0.5
        )
        status = "REJECTED" if rejected else "ACCEPTED"
        print(f"{img_path.name:30s} → {pred:10s} ({conf:.2f}) [{status}]")


if __name__ == "__main__":
    test_rejection_mechanisms()
    test_pipeline_on_images()
