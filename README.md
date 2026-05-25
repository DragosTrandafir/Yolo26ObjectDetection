# 🌸 Flower Object Detection with YOLO — Real vs. Synthetic Data

A computer vision project comparing object detection performance when training on **real data only** versus **real + synthetic data**, applied to a 14-class flower detection task using YOLO.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Synthetic Data Pipeline](#synthetic-data-pipeline)
- [Model](#model)
- [Results](#results)
- [How to Run](#how-to-run)
- [Key Takeaways](#key-takeaways)

---

## Project Overview

This project tackles **multi-class flower detection** (14 classes) as a study in:

1. Building a dataset from multiple heterogeneous sources with different annotation formats.
2. Implementing object detection evaluation metrics (IoU, Precision, Recall, AP, mAP) **from scratch**.
3. Generating synthetic training data via a copy-paste composition pipeline.
4. Training and evaluating a YOLO detector, comparing performance with and without synthetic data.

The project was completed as part of a Computer Vision laboratory assignment.

---

## Repository Structure

```
.
├── object_detection_model_and_dataset_analysis.ipynb   # Dataset analysis, metrics implementation, and case studies
├── yolo26_training.ipynb                               # YOLO training on real data only
├── yolo26_training_synthetic.ipynb                     # YOLO training on real + synthetic data
├── Normal_final_results_analysis.txt                   # Val/test metrics — real data model
├── Synthetic_final_results_analysis.txt                # Val/test metrics — synthetic data model
└── README.md
```

---

## Dataset

The dataset was assembled from **multiple public sources** and unified into YOLO format.

| Property | Value |
|---|---|
| Number of classes | 14 |
| Annotation format | YOLO (normalized `[class x_center y_center width height]`) |
| Splits | Train / Validation / Test |


**Dataset analysis** (see `object_detection_model_and_dataset_analysis.ipynb`) includes:
- Bounding box count per class (class imbalance visualization)
- Bounding box width, height, and area distributions (absolute and relative to image size)

Notable imbalances exist in the dataset: classes 12 and 13 are significantly more represented than others such as class 5, which has very few samples.

---

## Evaluation Metrics

All metrics were **implemented from scratch** in `object_detection_model_and_dataset_analysis.ipynb`, without relying on external libraries.

### Implemented metrics

**Intersection over Union (IoU)** — measures spatial overlap between a predicted box and a ground-truth box. Used to decide whether a prediction counts as a True Positive given a threshold τ.

**Precision & Recall** — computed after matching predictions to ground-truth boxes via IoU:
- *Precision*: of all predictions made, how many were correct?
- *Recall*: of all real objects, how many were found?

**Average Precision (AP)** — area under the Precision-Recall curve for a single class at a given IoU threshold. Computed by:
1. Sorting predictions by confidence score (descending)
2. Greedily matching each prediction to the best unmatched ground-truth box
3. Building the cumulative precision-recall curve
4. Applying precision envelope correction (monotonic pass from right to left)
5. Computing the area via the trapezoidal rule

**mean Average Precision (mAP@0.5)** — average of per-class AP values at IoU threshold 0.5. This is the primary comparison metric throughout the project.

The notebook also applies these metrics to **three provided case studies**, reasoning about failure modes (e.g. overconfident detectors, poor localization, class imbalance effects) and proposing concrete improvements.

---

## Synthetic Data Pipeline

To test whether augmenting with synthetic data improves generalization, a **Copy-Paste Composition** pipeline was used:

- Flower instances are extracted from existing images using their bounding boxes (with optional SAM-based segmentation masks).
- Extracted instances are pasted onto diverse background images.
- Placement, scale, and orientation are varied to produce realistic composites.
- Annotations are automatically generated for the new images.

This avoids the cost of generative/diffusion models while still increasing sample diversity, especially for underrepresented classes.

---

## Model

**YOLO** (You Only Look Once) was used for all experiments. Training was tracked using [Weights & Biases (wandb)](https://wandb.ai).

Two training runs were performed:
- `yolo26_training.ipynb` — trained on **real data only**
- `yolo26_training_synthetic.ipynb` — trained on **real + synthetic data**

Both runs use the same architecture and hyperparameters; the only difference is the training set composition.

---

## Results

All metrics computed at **mAP@IoU=0.5**.

### Validation Set

| Metric | Real Only | Real + Synthetic |
|---|---|---|
| mAP@0.5 | **0.4411** | 0.4255 |
| Precision | **0.4783** | 0.4215 |
| Recall | 0.5289 | **0.5556** |
| F1 | 0.5023 | **0.4793** |

### Test Set

| Metric | Real Only | Real + Synthetic |
|---|---|---|
| mAP@0.5 | **0.5117** | 0.4617 |
| Precision | **0.4948** | 0.4821 |
| Recall | 0.5644 | **0.5975** |
| F1 | 0.5273 | **0.5336** |

### Per-Class AP — Test Set

| Class | Real Only | Real + Synthetic |
|---|---|---|
| class_0 | 0.3785 | **0.4641** |
| class_1 | **0.7247** | 0.6901 |
| class_2 | 0.2422 | **0.2648** |
| class_3 | **0.3701** | 0.2967 |
| class_4 | 0.5780 | **0.6258** |
| class_5 | **1.0000** | 0.2500 |
| class_6 | 0.3282 | **0.3376** |
| class_7 | 0.3640 | **0.3758** |
| class_8 | **0.4400** | 0.3225 |
| class_9 | 0.3951 | **0.4332** |
| class_10 | **0.5821** | 0.6541* |
| class_11 | **0.6639** | 0.6511 |
| class_12 | 0.5429 | **0.5450** |
| class_13 | 0.5537 | **0.5527** |

### Analysis

- The **real-only model** achieves higher mAP and Precision overall, driven partly by the dominance of classes 12 and 13 in the dataset, and by class 5 reaching a perfect AP of 1.0 — an artifact of having too few test samples (overconfident detection, not true generalization).
- The **synthetic model** achieves higher Recall and F1 on the test set, and shows a notably **more balanced per-class AP distribution**, indicating broader generalization across the full label space.
- The synthetic model's training curves (wandb) are **more stable** across both loss and accuracy metrics, suggesting a more efficient learning trajectory and stronger potential for improvement with further training.
- The real-only model may be limited by data scarcity on rare classes, making it difficult to improve beyond a local optimum.

---

## How to Run

### Prerequisites

```bash
pip install ultralytics wandb
```

### Training

Open and run `yolo26_training.ipynb` for the real-data experiment, or `yolo26_training_synthetic.ipynb` for the real + synthetic experiment. Both notebooks include dataset preparation, training configuration, and evaluation steps.

### Metrics & Analysis

Open `object_detection_model_and_dataset_analysis.ipynb` for the from-scratch metrics implementation, dataset visualizations, and case study analyses.

---

## Key Takeaways

- **Class imbalance significantly impacts mAP** — a single dominant class (class 5 with 5 samples achieving AP=1.0) can inflate aggregate metrics while hiding generalization failures.
- **Synthetic data improves recall and balance** at the cost of some raw precision, suggesting its primary benefit is helping the model see more of the class distribution rather than sharpening confidence calibration.
- **Stable training curves are a positive signal** — the synthetic model's smoother loss/accuracy evolution across all logged classes indicates it is learning more uniformly and may benefit more from extended training.
- **mAP alone is not enough** — visual inspection of predictions alongside per-class AP breakdowns is essential to diagnose failure modes like overconfidence, missing small objects, or poor localization.
