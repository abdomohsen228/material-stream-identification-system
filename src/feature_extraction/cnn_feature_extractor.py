# src/feature_extraction/cnn_feature_extractor.py
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

# Load pretrained ResNet50 as feature extractor
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
cnn_model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_cnn_features(img_path):
    """
    Extract 2048-dim deep CNN features from a single image using ResNet50.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    features = cnn_model.predict(img, verbose=0)
    return features.flatten()
