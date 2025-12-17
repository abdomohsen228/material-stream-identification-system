import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from feature_extraction.cnn_feature_extractor_resnet import extract_cnn_features

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "augmented"
OUTPUT_CSV = ROOT / "data" / "cnn_features_resnet.csv"

data, labels, images = [], [], []

print("Starting CNN feature extraction (ResNet50)...")
print("Dataset:", DATASET)

for label in os.listdir(DATASET):
    class_dir = DATASET / label
    if not class_dir.is_dir():
        continue

    print(f"Processing class: {label}")

    img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not img_files:
        print(f"Warning: no images found in {class_dir}")
        continue

    for img_name in tqdm(img_files):
        img_path = class_dir / img_name
        try:
            features = extract_cnn_features(img_path)
            if features is not None and len(features) > 0:
                data.append(features)
                labels.append(label)
                images.append(img_name)
        except Exception as e:
            print("Error processing", img_path, e)

df = pd.DataFrame(data)
df['label'] = labels
df['image'] = images

df.to_csv(OUTPUT_CSV, index=False)
print("Feature extraction completed")
print("Saved to:", OUTPUT_CSV)
print("Shape:", df.shape)
