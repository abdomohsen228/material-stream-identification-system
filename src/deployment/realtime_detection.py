# src/deployment/realtime_detection.py
import cv2
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms, models
import sys

ROOT = Path(__file__).resolve().parents[2]  # points to project root
sys.path.append(str(ROOT / "src"))

from models.predict_with_rejection import predict_pipeline_with_rejection

MODEL_PATH = ROOT / "models" / "svm_resnet50_pipeline.pkl"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
pipeline = joblib.load(MODEL_PATH)
print(f"Loaded model pipeline from {MODEL_PATH}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash', 'unknown']
CONFIDENCE_THRESHOLD = 0.5  # Threshold for rejecting predictions as unknown
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet.fc = torch.nn.Identity()  # remove classification head
resnet.eval().to(DEVICE)
frame_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

@torch.no_grad()
def extract_features_from_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_t = frame_transform(img).unsqueeze(0).to(DEVICE)
    features = resnet(img_t)
    return features.cpu().numpy().flatten()

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
                # Use rejection mechanism to handle unknown class
                predicted_class, conf, rejected = predict_pipeline_with_rejection(
                    pipeline,
                    features.reshape(1, -1),
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                    unknown_class="unknown"
                )
                confidence = conf * 100
            # Overlay info
            frame_display = frame.copy()
            cv2.rectangle(frame_display, (10, 10), (420, 130), (0, 0, 0), -1)
            cv2.rectangle(frame_display, (10, 10), (420, 130), (255, 255, 255), 2)
            # Use red color for unknown class, green for known classes
            text_color = (0, 0, 255) if predicted_class == "unknown" else (0, 255, 0)
            cv2.putText(frame_display, f"Class: {predicted_class.upper()}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            cv2.putText(frame_display, f"Confidence: {confidence:.1f}%", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if predicted_class == "unknown":
                cv2.putText(frame_display, "[OUT-OF-DISTRIBUTION]", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Material Stream Identification", frame_display)
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
