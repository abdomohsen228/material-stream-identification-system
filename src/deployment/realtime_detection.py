import cv2
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "svm_resnet_pipeline.pkl"
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(224, 224, 3)
)
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
print("ResNet50 loaded")

print(f"Loading model from {MODEL_PATH}...")
if not MODEL_PATH.exists():
    MODEL_PATH = ROOT / "models" / "svm_cnn_pipeline.pkl"
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

pipeline = joblib.load(MODEL_PATH)
print("Model loaded")
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def extract_features_from_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = feature_extractor.predict(img, verbose=0)
    return features.flatten()
def run_realtime_detection(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")
    print("\n" + "="*60)
    print("Real-time Material Stream Identification")
    print("="*60)
    print("Press 'q' to quit")
    print("="*60 + "\n")
    frame_count = 0
    process_every_n = 5
    predicted_class = "Initializing..."
    confidence = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            if frame_count % process_every_n == 0:
                features = extract_features_from_frame(frame)
                prediction = pipeline.predict([features])[0]
                probabilities = pipeline.predict_proba([features])[0]
                confidence = np.max(probabilities) * 100
                predicted_class = prediction
            frame_display = frame.copy()
            cv2.rectangle(frame_display, (10, 10), (400, 100), (0, 0, 0), -1)
            cv2.rectangle(frame_display, (10, 10), (400, 100), (255, 255, 255), 2)
            text = f"Class: {predicted_class.upper()}"
            confidence_text = f"Confidence: {confidence:.1f}%"    
            cv2.putText(frame_display, text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_display, confidence_text, (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('Material Stream Identification', frame_display)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nCamera released. Goodbye!")
if __name__ == "__main__":
    run_realtime_detection()

