import sys
from pathlib import Path
import joblib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from feature_extraction.cnn_feature_extractor import extract_cnn_features


MODEL_DIR = Path("models")
TEST_DIR = Path("test_images")  


svm = joblib.load(MODEL_DIR / "svm_cnn_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
pca = joblib.load(MODEL_DIR / "pca.pkl")

print("Predicting unknown images...\n")

for img_path in TEST_DIR.iterdir():
    if not img_path.is_file():
        continue

    try:
        features = extract_cnn_features(img_path)

        # preprocessing
        features = scaler.transform([features])
        features = pca.transform(features)

        probs = svm.predict_proba(features)[0]
        pred_class = svm.classes_[np.argmax(probs)]
        confidence = np.max(probs)

        print(f"{img_path.name:30s} → {pred_class:10s} ({confidence*100:.2f}%)")

    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")
