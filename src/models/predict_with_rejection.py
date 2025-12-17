import sys
from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from models.rejection_mechanism import (
    SVMRejectionMechanism, 
    KNNRejectionMechanism,
    get_rejection_mechanism,
    DEFAULT_THRESHOLDS
)
def predict_svm_with_rejection(model, scaler, pca, features, 
                               confidence_threshold: float = 0.5,
                               unknown_class: str = 'unknown'):
    features_scaled = scaler.transform([features])
    features_pca = pca.transform(features_scaled)
    probabilities = model.predict_proba(features_pca)[0]
    classes = model.classes_
    rejection = SVMRejectionMechanism(
        confidence_threshold=confidence_threshold,
        unknown_class=unknown_class
    )  
    predicted_class, confidence, was_rejected = rejection.predict_with_proba(
        probabilities, classes
    )
    return predicted_class, confidence, was_rejected, probabilities
def predict_pipeline_with_rejection(pipeline, features,
                                    confidence_threshold: float = 0.5,
                                    unknown_class: str = 'unknown'):
    probabilities = pipeline.predict_proba([features])[0]
    classifier = pipeline.steps[-1][1]
    classes = classifier.classes_
    rejection = SVMRejectionMechanism(
        confidence_threshold=confidence_threshold,
        unknown_class=unknown_class
    )
    predicted_class, confidence, was_rejected = rejection.predict_with_proba(
        probabilities, classes
    )
    return predicted_class, confidence, was_rejected, probabilities
def predict_batch_with_rejection(pipeline, features_list,
                                 confidence_threshold: float = 0.5,
                                 unknown_class: str = 'unknown'):
    probabilities_array = pipeline.predict_proba(features_list)
    classifier = pipeline.steps[-1][1]
    classes = classifier.classes_
    rejection = SVMRejectionMechanism(
        confidence_threshold=confidence_threshold,
        unknown_class=unknown_class
    )
    results = rejection.predict_batch_with_proba(probabilities_array, classes)
    return results
def load_model_and_predict(image_path, model_path=None, 
                          confidence_threshold: float = 0.5,
                          model_type: str = 'pipeline'):
    from feature_extraction.cnn_feature_extractor import extract_cnn_features
    if model_path is None:
        model_path = ROOT / "models" / "svm_cnn_pipeline.pkl"
    if model_type == 'pipeline':
        pipeline = joblib.load(model_path)
    else:
        model_dir = ROOT / "models"
        model = joblib.load(model_dir / "svm_cnn_model.pkl")
        scaler = joblib.load(model_dir / "scaler.pkl")
        pca = joblib.load(model_dir / "pca.pkl")
        pipeline = None
    features = extract_cnn_features(image_path)
    if features is None:
        return 'unknown', 0.0, True
    if model_type == 'pipeline':
        pred_class, confidence, rejected, _ = predict_pipeline_with_rejection(
            pipeline, features, confidence_threshold
        )
    else:
        pred_class, confidence, rejected, _ = predict_svm_with_rejection(
            model, scaler, pca, features, confidence_threshold
        )
    return pred_class, confidence, rejected
