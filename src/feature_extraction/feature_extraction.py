# src/feature_extraction/feature_extraction.py

import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "augmented"
OUTPUT_CSV = ROOT / "data" / "extracted_features.csv"

# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ------------------------------------------------------------------
# Load pretrained ResNet50 (ImageNet)
# ------------------------------------------------------------------
print("Loading pretrained ResNet50...")

resnet = models.resnet50(
    weights=models.ResNet50_Weights.IMAGENET1K_V2
)

# Freeze ALL parameters (important)
for param in resnet.parameters():
    param.requires_grad = False

# Remove classification head → feature extractor only
resnet.fc = nn.Identity()

resnet.eval()
resnet.to(DEVICE)

# ------------------------------------------------------------------
# Image preprocessing
# (Improved over CenterCrop)
# ------------------------------------------------------------------
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ------------------------------------------------------------------
# Feature extraction function
# ------------------------------------------------------------------
@torch.no_grad()
def extract_resnet50_features(image_path, n_crops=5):
    img = Image.open(image_path).convert("RGB")
    feats = []

    for _ in range(n_crops):
        crop = transform(img).unsqueeze(0).to(DEVICE)
        f = resnet(crop)
        feats.append(f.cpu().numpy())

    feats = torch.tensor(feats).mean(dim=0)
    return feats.numpy().flatten()
# ------------------------------------------------------------------
# Run feature extraction
# ------------------------------------------------------------------
features = []
labels = []
images = []

print("Starting ResNet50 feature extraction")

for label in sorted(os.listdir(DATASET)):
    class_dir = DATASET / label
    if not class_dir.is_dir():
        continue

    print(f"Processing class: {label}")

    for img_name in tqdm(os.listdir(class_dir)):
        if not img_name.lower().endswith((".jpg", ".png")):
            continue

        img_path = class_dir / img_name

        try:
            feat = extract_resnet50_features(img_path)
            features.append(feat)
            labels.append(label)
            images.append(img_name)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# ------------------------------------------------------------------
# Save to CSV
# ------------------------------------------------------------------
df = pd.DataFrame(features)
df["label"] = labels
df["image"] = images

df.to_csv(OUTPUT_CSV, index=False)

print("\nFeature extraction completed")
print("Saved to:", OUTPUT_CSV)
print("Shape:", df.shape)
