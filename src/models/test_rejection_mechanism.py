
import sys
from pathlib import Path
import joblib
import numpy as np
# Add src to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
from models.rejection_mechanism import (
    SVMRejectionMechanism,
    KNNRejectionMechanism,
    ConfidenceBasedRejection,
    DEFAULT_THRESHOLDS
)
from feature_extraction.cnn_feature_extractor import extract_cnn_features
from models.predict_with_rejection import predict_svm_with_rejection, predict_pipeline_with_rejection
def test_svm_rejection():
    print("=" * 70)
    print("Testing SVM Rejection Mechanism")
    print("=" * 70)
    model_dir = ROOT / "models"
    try:
        svm = joblib.load(model_dir / "svm_cnn_model.pkl")
        scaler = joblib.load(model_dir / "scaler.pkl")
        pca = joblib.load(model_dir / "pca.pkl")
    except FileNotFoundError as e:
        print(f"Error: Model files not found. {e}")
        return
    test_image = ROOT / "test_images"
    if not test_image.exists():
        print("No test images found. Skipping SVM rejection test.")
        return 
    test_files = list(test_image.glob("*.jpg"))[:3]
    for threshold_name, threshold_value in DEFAULT_THRESHOLDS.items():
        print(f"\n--- Testing with {threshold_name} threshold: {threshold_value} ---")
        for img_path in test_files:
            try:
                features = extract_cnn_features(img_path)
                if features is None:
                    continue
                pred_class, confidence, rejected, _ = predict_svm_with_rejection(
                    svm, scaler, pca, features,
                    confidence_threshold=threshold_value
                )
                status = "REJECTED → unknown" if rejected else "ACCEPTED"
                print(f"{img_path.name:30s} → {pred_class:10s} ({confidence:.3f}) [{status}]")
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
def test_rejection_mechanisms():
    classes = np.array(['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'])
    high_conf_probs = np.array([0.85, 0.05, 0.03, 0.02, 0.03, 0.02])
    low_conf_probs = np.array([0.35, 0.20, 0.15, 0.12, 0.10, 0.08])
    borderline_probs = np.array([0.55, 0.20, 0.10, 0.08, 0.05, 0.02])
    test_cases = [
        ("High Confidence", high_conf_probs),
        ("Low Confidence", low_conf_probs),
        ("Borderline", borderline_probs)
    ]
    rejection = SVMRejectionMechanism(confidence_threshold=0.5, unknown_class='unknown')
    print("\nConfidence Threshold: 0.5\n")
    for test_name, probs in test_cases:
        pred_class, confidence, rejected = rejection.predict_with_proba(probs, classes)
        status = "REJECTED → unknown" if rejected else "ACCEPTED"
        print(f"{test_name:20s} → Predicted: {pred_class:10s} "
              f"Confidence: {confidence:.3f} [{status}]")
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Rejection Mechanism Test Suite")
    print("=" * 70)
    test_rejection_mechanisms()
    test_svm_rejection()
