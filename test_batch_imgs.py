# test_batch_imgs.py
import numpy as np
import joblib
from pathlib import Path
from feature_extraction.cnn_feature_extractor import extract_cnn_features

MODEL_PATH = Path('models/svm_cnn_pipeline.pkl')
pipeline = joblib.load(MODEL_PATH)

BATCH_DIR = Path('data/test_batch')
img_paths = [f for f in BATCH_DIR.glob('*') if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]

X_test, img_names = [], []

for img_path in img_paths:
    try:
        features = extract_cnn_features(img_path)
        if features is not None and len(features) > 0:
            X_test.append(features)
            img_names.append(img_path.name)
    except Exception as e:
        print("Error:", img_path, e)

if len(X_test) == 0:
    raise ValueError("No valid features found for batch images!")

X_test = np.array(X_test)
if X_test.ndim == 1:
    X_test = X_test.reshape(1, -1)

y_pred = pipeline.predict(X_test)

for name, pred in zip(img_names, y_pred):
    print(name, "->", pred)
