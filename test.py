import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from collections import Counter

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from feature_extraction.feature_extraction import extract_resnet50_features
from models.predict_with_rejection import (
    predict_pipeline_with_rejection,
    predict_batch_with_rejection
)

MODEL_PATH = ROOT / "models" / "svm_resnet50_pipeline.pkl"
OUTPUT_EXCEL = ROOT / "predictions.xlsx"

pipeline = joblib.load(MODEL_PATH)


def predict_folder(folder_path, confidence_threshold=0.5):
    features_list = []
    image_names = []

    for img_path in Path(folder_path).iterdir():
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        features = extract_resnet50_features(img_path)
        features_list.append(features)
        image_names.append(img_path.name)

    predictions = predict_batch_with_rejection(
        pipeline,
        np.array(features_list),
        confidence_threshold=confidence_threshold,
        unknown_class="unknown"
    )

    return image_names, predictions


def save_results_to_excel(image_names, predictions, output_path):
    rows = []

    for img_name, (label, confidence, rejected) in zip(image_names, predictions):
        rows.append({
            "ImageName": img_name,
            "predictedlabel": "unknown" if rejected else label
        })

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)


def print_statistics(predictions):
    labels = []
    unknown_count = 0

    for label, confidence, rejected in predictions:
        if rejected:
            unknown_count += 1
            labels.append("unknown")
        else:
            labels.append(label)

    counter = Counter(labels)

    print("\nPrediction Summary")
    print("-" * 30)
    print(f"Total images: {len(predictions)}")
    print(f"Unknown predictions: {unknown_count}\n")

    print("Class distribution:")
    for cls, count in counter.items():
        print(f"- {cls:<10} : {count}")


if __name__ == "__main__":
    print("\nRunning test.py demo\n")
    test_images_dir = ROOT / "test_images"
    image_names, results = predict_folder(
        test_images_dir,
        confidence_threshold=0.5
    )

    save_results_to_excel(
        image_names,
        results,
        OUTPUT_EXCEL
    )

    print_statistics(results)

    print(f"\nExcel file saved to: {OUTPUT_EXCEL}")
