import numpy as np
from models.rejection_mechanism import SVMRejectionMechanism

def predict_pipeline_with_rejection(
    pipeline,
    features,
    confidence_threshold: float = 0.5,
    unknown_class: str = "unknown"
):
    probabilities = pipeline.predict_proba(features)[0]
    classifier = pipeline.steps[-1][1]
    classes = classifier.classes_

    rejection = SVMRejectionMechanism(
        confidence_threshold=confidence_threshold,
        unknown_class=unknown_class
    )

    return rejection.predict_with_proba(probabilities, classes)


def predict_batch_with_rejection(
    pipeline,
    features_list,
    confidence_threshold: float = 0.5,
    unknown_class: str = "unknown"
):
    probabilities_array = pipeline.predict_proba(features_list)
    classifier = pipeline.steps[-1][1]
    classes = classifier.classes_

    rejection = SVMRejectionMechanism(
        confidence_threshold=confidence_threshold,
        unknown_class=unknown_class
    )
    return rejection.predict_batch_with_proba(probabilities_array, classes)
