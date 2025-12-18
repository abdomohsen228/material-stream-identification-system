import sys
from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from feature_extraction.feature_extraction import extract_resnet50_features
from models.predict_with_rejection import (
    predict_pipeline_with_rejection,
    predict_batch_with_rejection
)

MODEL_PATH = ROOT / "models" / "svm_resnet50_pipeline.pkl"

pipeline = joblib.load(MODEL_PATH)

def predict_single_image(image_path, confidence_threshold=0.5):
    features = extract_resnet50_features(image_path).reshape(1, -1)
    return predict_pipeline_with_rejection(
        pipeline,
        features,
        confidence_threshold=confidence_threshold,
        unknown_class="unknown"
    )

def predict_folder(folder_path, confidence_threshold=0.5):
    results = {}
    features_list = []
    image_names = []

    for img_path in Path(folder_path).iterdir():
        if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue
        features = extract_resnet50_features(img_path)
        features_list.append(features)
        image_names.append(img_path.name)

    preds = predict_batch_with_rejection(
        pipeline,
        np.array(features_list),
        confidence_threshold=confidence_threshold,
        unknown_class="unknown"
    )

    for name, result in zip(image_names, preds):
        results[name] = result
    return results

if __name__ == "__main__":
    print("\nRunning test.py demo\n")
    test_images_dir = ROOT / "test_images"
    results = predict_folder(test_images_dir, confidence_threshold=0.5)
    for img, (pred, conf, rejected) in results.items():
        status = "REJECTED" if rejected else "ACCEPTED"
        print(f"{img:30s} → {pred:10s} ({conf*100:.2f}%) [{status}]")
