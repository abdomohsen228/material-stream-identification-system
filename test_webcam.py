import cv2
import numpy as np
import joblib
from pathlib import Path
from src.feature_extraction.feature_extraction import extract_resnet50_features
from src.models.predict_with_rejection import predict_batch_with_rejection 

# --------------------------------------------------
# Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # adjust if needed
MODEL_PATH = ROOT / "models" / "svm_resnet50_pipeline.pkl"

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
pipeline = joblib.load(MODEL_PATH)
CONFIDENCE_THRESHOLD = 0.5
UNKNOWN_CLASS = "unknown"

# --------------------------------------------------
# Open webcam
# --------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Resize or crop to 224x224 to match ResNet input
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Save temp image to pass to extract_resnet50_features
    # Or modify extract_resnet50_features to accept numpy array
    # Here we'll save to a temp file
    import tempfile
    import os
    temp_path = os.path.join(tempfile.gettempdir(), "tmp_cam.jpg")
    cv2.imwrite(temp_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    # Extract features
    features = extract_resnet50_features(temp_path).reshape(1, -1)

    # Predict with rejection
    pred, conf, rejected = predict_with_rejection(
        pipeline, features,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        unknown_class=UNKNOWN_CLASS
    )

    status = "REJECTED" if rejected else "ACCEPTED"

    # Overlay prediction on frame
    label_text = f"{pred} ({conf*100:.1f}%) {status}"
    cv2.putText(frame, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if not rejected else (0, 0, 255), 2)

    cv2.imshow("Real-Time Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
