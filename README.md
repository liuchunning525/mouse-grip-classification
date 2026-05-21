# Mouse Grip Classification and Personalized Mouse Recommendation

A computer vision and machine learning project for recognizing mouse grip styles and analyzing natural grip tendency for personalized mouse recommendation.

This project combines hand tracking, ArUco marker tracking, mouse movement logs, and machine learning to classify standard mouse grips and estimate users' natural grip tendencies.

---

## Overview

The goal of this project is to recognize mouse grip styles and explore whether grip recognition can support personalized mouse recommendation.

The system analyzes:

- Hand landmarks from video
- Mouse movement behavior
- Click task performance
- Standard grip classification
- Free-use grip tendency
- Prototype mouse recommendation

The main grip classes are:

- Palm
- Claw
- Fingertip
---

## Dataset

The latest dataset contains:

| Item | Value |
|---|---:|
| Users | 20 |
| Mice | G102, X2H, XliteV3ES |
| Conditions | A, B, C, D |
| Samples | 7837 |
| Features | 888 |
| Classes | palm, claw, fingertip |

Class distribution:

| Class | Samples |
|---|---:|
| Palm | 2643 |
| Fingertip | 2691 |
| Claw | 2503 |

---

## Data Collection

Each participant performed target-clicking tasks using three different mice under four target conditions.

The collected data includes:

- Webcam video
- Task log JSON
- Mouse trajectory
- Click timing
- Target condition information

---

## Feature Extraction

The system extracts both hand posture and mouse interaction features.

### Hand Features

- 21 MediaPipe hand landmarks
- Finger curl features
- Joint angle features
- Finger extension features
- Finger spread features
- Palm-related structure features
- Hand compactness features

### Mouse Interaction Features

- ArUco marker position
- Mouse-relative hand features
- Mouse movement speed
- Path length
- Peak speed
- Movement stability
- Trial-level temporal statistics

The final feature vector contains 888 features.

---

## Model

The main classification model is:

```text
Random Forest Classifier
```

The model predicts:

```text
claw / fingertip / palm
```
---

## Evaluation

The model is evaluated using:

```text
Leave-One-User-Out Cross Validation
```

This evaluation tests whether the model can recognize grip styles from an unseen user.

### Latest Result

| Metric | Value |
|---|---:|
| Overall Accuracy | 0.7918 |
| Samples | 7837 |
| Features | 888 |
| Users | 20 |

### Confusion Matrix

| Actual \\ Predicted | Claw | Fingertip | Palm |
|---|---:|---:|---:|
| Claw | 1861 | 507 | 135 |
| Fingertip | 413 | 2070 | 208 |
| Palm | 118 | 251 | 2274 |

Palm grip was recognized most stably, while claw and fingertip still showed noticeable confusion for some users.

---

## Free Grip Tendency Analysis

Natural mouse grip is often not a strict single class. Therefore, the free grip analysis reports a probability-based tendency instead of only one label.

Example output:

```text
user_020:
claw=0.36
fingertip=0.42
palm=0.22
tendency=fingertip + claw
recommended=X2H
```

The system currently distinguishes:

- Palm dominant
- Claw dominant
- Fingertip dominant
- Palm + Claw hybrid
- Claw + Fingertip hybrid

Palm + Fingertip hybrid is treated as unreliable or uncommon and is converted to the dominant class.

---

## Latest Free Grip Summary

| User | Claw | Fingertip | Palm | Tendency | Recommended Mouse |
|---|---:|---:|---:|---|---|
| user_001 | 0.22 | 0.47 | 0.31 | fingertip | XliteV3ES |
| user_002 | 0.08 | 0.06 | 0.86 | palm | G102 |
| user_003 | 0.11 | 0.27 | 0.62 | palm | G102 |
| user_004 | 0.36 | 0.31 | 0.34 | claw + palm | G102 |
| user_005 | 0.30 | 0.17 | 0.53 | palm | G102 |
| user_006 | 0.06 | 0.15 | 0.79 | palm | G102 |
| user_007 | 0.16 | 0.30 | 0.53 | palm | G102 |
| user_008 | 0.12 | 0.08 | 0.81 | palm | G102 |
| user_009 | 0.44 | 0.26 | 0.30 | claw + palm | G102 |
| user_010 | 0.01 | 0.01 | 0.99 | palm | G102 |
| user_011 | 0.01 | 0.02 | 0.96 | palm | G102 |
| user_012 | 0.10 | 0.10 | 0.80 | palm | G102 |
| user_013 | 0.36 | 0.09 | 0.55 | palm | G102 |
| user_014 | 0.01 | 0.01 | 0.97 | palm | G102 |
| user_015 | 0.33 | 0.46 | 0.21 | fingertip + claw | X2H |
| user_016 | 0.33 | 0.34 | 0.33 | fingertip | XliteV3ES |
| user_017 | 0.16 | 0.18 | 0.66 | palm | G102 |
| user_018 | 0.24 | 0.63 | 0.13 | fingertip | XliteV3ES |
| user_019 | 0.38 | 0.24 | 0.37 | claw + palm | G102 |
| user_020 | 0.36 | 0.42 | 0.22 | fingertip + claw | X2H |

---

## Prototype Recommendation Rule

The current prototype recommendation rule is:

| Grip Tendency | Recommended Mouse |
|---|---|
| Palm | G102 |
| Claw | X2H |
| Fingertip | XliteV3ES |
| Palm + Claw | G102 |
| Claw + Fingertip | X2H |

This recommendation rule is still a prototype and should be validated using:

- Task performance
- Movement efficiency
- User comfort questionnaire
- Subjective preference feedback

---

## Pipeline

```text
Video Recording
    ↓
WebM to MP4 Conversion
    ↓
ArUco Marker Detection
    ↓
MediaPipe Hand Landmark Extraction
    ↓
Frame-Level Feature Extraction
    ↓
Trial-Level Feature Aggregation
    ↓
Training Dataset Construction
    ↓
Random Forest Training
    ↓
Leave-One-User-Out Evaluation
    ↓
Free Grip Tendency Prediction
    ↓
Prototype Mouse Recommendation
    ↓
Performance and Questionnaire Validation
```

---

## Main Scripts

| Script | Description |
|---|---|
| `detect_aruco_mouse_ref.py` | Detect reference ArUco marker pose for each mouse |
| `extract_video_features_aruco.py` | Extract hand and mouse features from video |
| `aggregate_trial_features_v2.py` | Aggregate frame-level data into trial-level features |
| `build_training_dataset_3mouse_v2_relative_speedfilter.py` | Build training dataset |
| `train_grip_model_3mouse_eval.py` | Train and evaluate Random Forest model |
| `predict_free_grip.py` | Predict natural grip tendency and prototype recommendation |
| `analyze_free_mouse_performance.py` | Analyze free-use mouse performance |

---

## Example Commands

### Train and Evaluate

```bash
python src/train_grip_model_3mouse_eval.py --input data/datasets/training_dataset_3mouse_20users.json --mode leave_one_user --report_out data/datasets/report_20users_leave_one_user.json
```

### Predict Free Grip Tendency

```bash
python src/predict_free_grip.py --input_root test/outputs --model models/trained/grip_model_3mouse.pkl --feature_columns models/trained/feature_columns_3mouse.json --label_encoder models/trained/label_encoder_3mouse.json --output data/datasets/free_prediction_report.json
```

### Analyze Mouse Performance

```bash
python src/analyze_free_mouse_performance.py --raw_root test/raw --prediction_report data/datasets/free_prediction_report.json --output data/datasets/free_mouse_performance_report.json
```

---

## Repository Notice

The full raw dataset is not included in this repository because it contains large video files and user-related data.

The following folders are recommended to be excluded from GitHub:

```text
data/raw/
data/outputs/
test/raw/
test/outputs/
models/trained/
```

Only source code, configuration files, documentation, and small sample data should be uploaded.

---

## Future Work

- Validate recommendation results using task performance
- Compare recommendation results with user questionnaires
- Add more mouse shapes
- Improve claw/fingertip separation
- Investigate real-time grip recognition

---

## Team

- Jialiang Jiang
- Chunning Liu
- Zeqi Yao
- Hanlin Wang

Yonsei University
