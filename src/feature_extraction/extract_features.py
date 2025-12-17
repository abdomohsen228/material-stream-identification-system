import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern
from skimage import exposure, img_as_ubyte
import mahotas
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from feature_extraction.cnn_feature_extractor import extract_cnn_features


# ==========================================
# Resolve project root dynamically
# ==========================================

ROOT = Path(__file__).resolve().parents[2]  # go back to project root

dataset_path = ROOT / "data" / "augmented"
output_csv   = ROOT / "data" / "full_features.csv"


# ============================
# FUNCTIONS
# ============================

def normalize_features(features):
    norm = np.linalg.norm(features)
    return features / (norm + 1e-7)

def extract_hog(img_rgb):
    fd, _ = hog(
        img_rgb,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        channel_axis=-1
    )
    return normalize_features(fd)

def extract_lbp(gray):
    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=int(lbp.max() + 1),
        range=(0, lbp.max() + 1),
        density=True
    )
    return normalize_features(hist)

def extract_haralick(gray):
    gray_uint8 = img_as_ubyte(gray)
    haralick = mahotas.features.haralick(gray_uint8).mean(axis=0)
    return normalize_features(haralick)

def extract_color_hist(img):
    hist_b = cv2.calcHist([img], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [32], [0, 256])

    hist_b = cv2.normalize(hist_b, hist_b).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_r = cv2.normalize(hist_r, hist_r).flatten()

    return normalize_features(np.concatenate([hist_b, hist_g, hist_r]))


def extract_combined_features(path):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (128, 128))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_feat = extract_hog(img_rgb) * 0.3
    lbp_feat = extract_lbp(gray) * 0.3
    haralick_feat = extract_haralick(gray) * 0.2
    color_feat = extract_color_hist(img) * 0.2

    return np.concatenate([hog_feat, lbp_feat, haralick_feat, color_feat])


# ============================
# MAIN EXTRACTION LOOP
# ============================

def main():

    print("📌 Starting dataset feature extraction...")
    print("📁 Dataset path:", dataset_path)

    data = []
    labels = []
    images = []

    for material in os.listdir(dataset_path):
        material_path = dataset_path / material
        if not material_path.is_dir():
            continue

        print(f"\n🔹 Processing class: {material}")

        for img_name in tqdm(os.listdir(material_path)):
            img_path = material_path / img_name

            try:
                features = extract_combined_features(img_path)
                data.append(features)
                labels.append(material)
                images.append(img_name)

            except Exception as e:
                print("⚠️ Error reading:", img_path, "|", e)


    df = pd.DataFrame(data)
    df["label"] = labels
    df["image"] = images
    df.to_csv(output_csv, index=False)

    print("\n✅ Feature extraction completed!")
    print("📁 Saved to:", output_csv)
    print("📊 Dataset shape:", df.shape)



if __name__ == "__main__":
    main()
