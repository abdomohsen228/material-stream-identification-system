import numpy as np

def predict_with_rejection(pipeline, X, confidence_threshold=0.5, unknown_class="unknown"):
    probs = pipeline.predict_proba(X)
    max_probs = probs.max(axis=1)
    preds = pipeline.classes_[probs.argmax(axis=1)]

    rejected = max_probs < confidence_threshold
    preds = np.where(rejected, unknown_class, preds)

    return preds[0], max_probs[0], rejected[0]


def predict_batch_with_rejection(pipeline, X, confidence_threshold=0.5, unknown_class="unknown"):
    probs = pipeline.predict_proba(X)
    max_probs = probs.max(axis=1)
    preds = pipeline.classes_[probs.argmax(axis=1)]

    results = []
    for p, c in zip(preds, max_probs):
        if c < confidence_threshold:
            results.append((unknown_class, c, True))
        else:
            results.append((p, c, False))
    return results
