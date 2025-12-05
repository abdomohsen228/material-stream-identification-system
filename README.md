# Material Stream Identification System (MSI)

Machine Learning system for classifying waste materials into seven categories using feature extraction, SVM and k-NN classifiers, and real-time camera processing.

## Dataset

7 classes: **glass**, **paper**, **cardboard**, **plastic**, **metal**, **trash**, **unknown**

## Project Structure

```
material-stream-identification-system/
├── data/
│   ├── raw/              # Original dataset images
│   ├── augmented/        # Augmented images (+30%)
│   └── prepared/         # Processed images and features
├── src/
│   ├── data_preparation/ # Data preprocessing
│   ├── feature_extraction/ # HOG, LBP, Haralick, color histograms
│   ├── models/           # SVM and k-NN classifiers
│   └── deployment/       # Real-time OpenCV camera processing
├── notebooks/            # Jupyter notebooks
├── reports/              # Results and visualizations
├── saved_models/         # Trained models
├── requirements.txt
└── README.md
```

## Installation

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Complete Workflow

1. **Data Preparation:**
   ```bash
   python src/data_preparation/prepare_data.py
   ```

2. **Data Augmentation (+30%):**
   ```bash
   python src/data_preparation/augment_data.py
   ```

3. **Feature Extraction:**
   ```bash
   python src/feature_extraction/extract_features.py
   ```
   Extracts: HOG, LBP, Haralick textures, color histograms

4. **Train Models:**
   ```bash
   python src/models/train_models.py
   ```
   Trains SVM and k-NN classifiers

5. **Real-Time Detection:**
   ```bash
   python src/deployment/realtime_detection.py
   ```

## Requirements

See `requirements.txt` for full list. Main packages:
- numpy, pandas, opencv-python, scikit-learn, scikit-image
- Pillow, imgaug, mahotas, matplotlib, tqdm, joblib, jupyter
