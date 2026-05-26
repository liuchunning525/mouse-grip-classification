# Mouse Grip Classification and Personalized Mouse Recommendation

A computer vision and machine learning project for recognizing mouse grip styles, estimating natural grip tendency, and validating personalized mouse recommendation.

---

# Overview

This project combines:

* MediaPipe hand landmark tracking
* ArUco marker-based mouse tracking
* Mouse movement analysis
* Machine learning classification
* Free-grip tendency estimation
* Personalized mouse recommendation
* Recommendation validation using task performance and questionnaires

The system recognizes the following standard grip classes:

* Palm
* Claw
* Fingertip

---

# Dataset
## Standard Grip Dataset

| Item       |                   Value |
| ---------- | ----------------------: |
| Users      |                      20 |
| Conditions |           A / B / C / D |
| Samples    |                    7837 |
| Features   |                     888 |
| Classes    | palm / claw / fingertip |

## Class Distribution
| Class     | Samples |
| --------- | ------: |
| Palm      |    2643 |
| Fingertip |    2691 |
| Claw      |    2503 |

---

# Data Collection

Each participant performed target-clicking tasks using multiple mice under standardized experimental conditions.

Collected data includes:

* Webcam video
* Mouse trajectory
* Click timing logs
* Target condition information
* Free-use grip recording

---
# Feature Extraction
## Hand Features

* 21 MediaPipe hand landmarks
* Finger curl features
* Finger extension features
* Joint angle features
* Finger spread features
* Palm structure features
* Hand compactness features

## Mouse Interaction Features

* ArUco marker pose
* Relative hand-to-mouse position
* Mouse movement speed
* Path length
* Movement smoothness
* Peak speed
* Trial-level temporal statistics

---

# Model

## Classification Model

Random Forest Classifier

The model predicts:  palm / claw / fingertip

---

# Evaluation

The model is evaluated using:

```
Leave-One-User-Out Cross Validation
```

This evaluation measures whether the system can recognize grip styles from unseen users.

## Latest Result
| Metric           |  Value |
| ---------------- | -----: |
| Overall Accuracy | 0.7918 |
| Samples          |   7837 |
| Features         |    888 |
| Users            |     20 |

## Confusion Matrix
| Actual \ Predicted | Claw | Fingertip | Palm |
| ------------------ | ---: | --------: | ---: |
| Claw               | 1861 |       507 |  135 |
| Fingertip          |  413 |      2070 |  208 |
| Palm               |  118 |       251 | 2274 |

Palm grip was recognized most stably, while claw and fingertip still showed partial confusion for some users.

---

# Free Grip Tendency Analysis

The system estimates probability-based grip tendency instead of a strict single-class label.

## Current Tendency Types

* Palm dominant
* Claw dominant
* Fingertip dominant
* Palm + Claw hybrid
* Claw + Fingertip hybrid

Palm + Fingertip hybrid is treated as unreliable and converted to the dominant class.

---

# Recommendation System
## Baseline Mouse
G102

G102 is used only for:

- Free-grip recording
- Natural grip tendency estimation

It is NOT used as a recommendation target.

## Recommendation Candidates

| Grip Tendency    | Recommended Mouse       |
| ---------------- | ----------------------- |
| Palm             | XliteCrazyLight         |
| Claw             | X2H                     |
| Fingertip        | XliteV3ES               |
| Palm + Claw      | Dominant tendency mouse |
| Claw + Fingertip | Dominant tendency mouse |

Hybrid tendencies follow the stronger tendency.

# Free Grip Prediction Summary

| Tendency                | Users |
| ----------------------- | ----: |
| Palm dominant           |    15 |
| Fingertip dominant      |     3 |
| Palm + Claw hybrid      |     2 |
| Claw dominant           |     0 |
| Claw + Fingertip hybrid |     0 |

Most users showed palm-related tendencies under natural free-grip conditions.

# Recommendation Validation

The recommendation system is validated using both:

## Objective Evaluation

- Completion time
- Click accuracy
- Movement efficiency
- Overshoot stability

## Subjective Evaluation

- Comfort
- Preference
- Fatigue
- Perceived control quality

---

# Recommendation Validation Result

The system compares:

recommended mouse
vs
non-recommended mice


using the same task condition.

## Current Result

| Metric                  | Value |
| ----------------------- | ----: |
| Users Tested            |    20 |
| Recommendation Success  |    14 |
| Recommendation Accuracy | 0.700 |

A recommendation is considered successful when the recommended mouse achieves the best performance score among candidate mice.

## Recommendation Performance Improvement

Among users with successful recommendations,
the recommended mouse improved task performance by an average of 16.61% compared to non-recommended mice.

---

# Current Research Pipeline

```text
Data Collection
    ↓
Feature Extraction
    ↓
Model Training
    ↓
Grip Tendency Prediction
    ↓
Mouse Recommendation
    ↓
Recommendation Validation
```

---

# Main Scripts

| Script                                                     | Description                                              |
| ---------------------------------------------------------- | -------------------------------------------------------- |
| `detect_aruco_mouse_ref.py`                                | Detect reference mouse pose using ArUco markers          |
| `extract_video_features_aruco.py`                          | Extract hand and mouse interaction features              |
| `aggregate_trial_features_v2.py`                           | Aggregate frame-level features into trial-level features |
| `build_training_dataset_3mouse_v2_relative_speedfilter.py` | Build the final training dataset                         |
| `train_grip_model_3mouse_eval.py`                          | Train and evaluate the Random Forest classifier          |
| `predict_free_grip.py`                                     | Predict natural grip tendency and recommend mouse        |
| `analyze_free_mouse_performance.py`                        | Compare free-use mouse performance                       |

---

# Example Commands

## 1. Train and Evaluate Model

```bash
python src/train_grip_model_3mouse_eval.py --input data/datasets/training_dataset_3mouse_20users.json --mode leave_one_user --report_out data/datasets/report_20users_leave_one_user.json
```

---

## 2. Predict Free Grip Tendency

```bash
python src/predict_free_grip.py --input_root test/outputs --baseline_mouse G102 --model models/trained/grip_model_3mouse.pkl --feature_columns models/trained/feature_columns_3mouse.json --label_encoder models/trained/label_encoder_3mouse.json --output data/datasets/free_prediction_report_g102_only.json
```

---

## 3. Analyze Recommendation Performance

```bash
python src/analyze_free_mouse_performance.py --raw_root test/raw --prediction_report data/datasets/free_prediction_report_g102_only.json --output data/datasets/free_mouse_performance_report.json
```

---

# Repository Notice

Large raw video datasets are not included in this repository.

Recommended exclusions for GitHub:

```text
/data/raw/
/data/outputs/
/test/raw/
/test/outputs/
/models/trained/
```

In addition, training_dataset_3mouse_20users.json is not uploaded because the file size is too large for the repository.

---

# Future Work

- Improve hybrid grip analysis
- Expand validation experiments
- Add more mouse shapes
- Explore real-time grip recognition

---

# Team

* Jialiang Jiang
* Chunning Liu
* Zeqi Yao
* Hanlin Wang


* Professor:이병주
* Assistant:이한별


Yonsei University
