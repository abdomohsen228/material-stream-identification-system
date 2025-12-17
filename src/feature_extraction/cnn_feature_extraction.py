# src/feature_extraction/cnn_feature_extraction.py
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
from imgaug import augmenters as iaa
import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from feature_extraction.cnn_feature_extractor import extract_cnn_features

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/augmented"
OUTPUT_CSV = ROOT / "data/cnn_features.csv"

# Augmentation pipeline
aug_pipeline = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.3),
    iaa.Sometimes(0.5, iaa.Affine(rotate=(-25,25), mode='reflect')),
    iaa.Sometimes(0.5, iaa.Multiply((0.8,1.2))),
    iaa.Sometimes(0.5, iaa.LinearContrast((0.8,1.2)))
])

data, labels, images = [], [], []

print("📌 Starting CNN feature extraction...")

for label in os.listdir(DATASET):
    class_dir = DATASET / label
    if not class_dir.is_dir():
        continue

    print(f"🔹 Processing class: {label}")
    img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg','.png'))]

    for img_name in tqdm(img_files):
        img_path = class_dir / img_name
        try:
            # Original features
            features = extract_cnn_features(img_path)
            data.append(features)
            labels.append(label)
            images.append(img_name)

            # Augmented features
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224,224))
            aug_img = aug_pipeline(image=img)
            aug_path = str(img_path)  # dummy path for logging
            features_aug = extract_cnn_features_from_array(aug_img)
            data.append(features_aug)
            labels.append(label)
            images.append(f"{img_name}_aug")
        except Exception as e:
            print("Error:", img_path, e)

df = pd.DataFrame(data)
df["label"] = labels
df["image"] = images
df.to_csv(OUTPUT_CSV, index=False)

print("Feature extraction completed")
print("Saved to:", OUTPUT_CSV)
print("Shape:", df.shape)

# Helper to extract features from numpy array (augmented image)
def extract_cnn_features_from_array(img_array):
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = cnn_model.predict(img_array, verbose=0)
    return features.flatten()
