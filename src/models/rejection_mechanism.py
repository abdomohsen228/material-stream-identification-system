import numpy as np
from typing import Tuple, Optional, Union


class RejectionMechanism:
    def __init__(self, confidence_threshold: float = 0.5, unknown_class: str = 'unknown'):
        self.confidence_threshold = confidence_threshold
        self.unknown_class = unknown_class

    def predict_with_rejection(self, prediction: str, confidence: float) -> Tuple[str, float, bool]:
        if confidence < self.confidence_threshold:
            return (self.unknown_class, confidence, True)
        return (prediction, confidence, False)


class SVMRejectionMechanism(RejectionMechanism):
    def __init__(self, confidence_threshold: float = 0.5, unknown_class: str = 'unknown'):
        super().__init__(confidence_threshold, unknown_class)
    
    def predict_with_proba(self, probabilities: np.ndarray, classes: np.ndarray) -> Tuple[str, float, bool]:
        predicted_idx = np.argmax(probabilities)
        predicted_class = classes[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        if confidence < self.confidence_threshold:
            return (self.unknown_class, confidence, True)
        
        return (predicted_class, confidence, False)
    
    def predict_batch_with_proba(self, probabilities_array: np.ndarray, classes: np.ndarray) -> list:
        results = []
        for probs in probabilities_array:
            result = self.predict_with_proba(probs, classes)
            results.append(result)
        return results


class KNNRejectionMechanism(RejectionMechanism):
    def __init__(self, confidence_threshold: float = 0.5, unknown_class: str = 'unknown',
                 use_distance: bool = True, distance_threshold: float = None):

        super().__init__(confidence_threshold, unknown_class)
        self.use_distance = use_distance
        self.distance_threshold = distance_threshold
    
    def predict_with_knn(self, prediction: str, confidence: float, 
                        distance: Optional[float] = None) -> Tuple[str, float, bool]:
        if self.use_distance and distance is not None and self.distance_threshold is not None:
            if distance > self.distance_threshold:
                return (self.unknown_class, confidence, True)
        
        if confidence < self.confidence_threshold:
            return (self.unknown_class, confidence, True)
        
        return (prediction, confidence, False)


class ConfidenceBasedRejection(RejectionMechanism):
    def __init__(self, confidence_threshold: float = 0.5, unknown_class: str = 'unknown'):
        super().__init__(confidence_threshold, unknown_class)
    
    def reject_if_uncertain(self, prediction: str, confidence: float) -> Tuple[str, float, bool]:
        return self.predict_with_rejection(prediction, confidence)


def get_rejection_mechanism(model_type: str = 'svm', **kwargs) -> RejectionMechanism:

    if model_type.lower() == 'svm':
        return SVMRejectionMechanism(**kwargs)
    elif model_type.lower() == 'knn':
        return KNNRejectionMechanism(**kwargs)
    else:
        return ConfidenceBasedRejection(**kwargs)
DEFAULT_THRESHOLDS = {
    'strict': 0.7,    # Only accept high-confidence predictions
    'moderate': 0.5,  # Balanced approach (default)
    'lenient': 0.3    # Accept most predictions
}

