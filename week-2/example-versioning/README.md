# Week 3 Assignment - Integrating Feast Feature Store into the IRIS Pipeline

## Overview

This assignment extends the Week 2 MLOps pipeline by integrating **Feast Feature Store** into the IRIS machine learning workflow.

In Week 2, DVC was used to version datasets and model artifacts. However, DVC alone does not guarantee that the same feature engineering logic is used during both model training and inference.

Feast solves this problem by acting as a centralized feature store that provides:

- Consistent feature definitions
- Offline feature retrieval for training
- Online feature retrieval for inference
- Elimination of training-serving skew

This implementation uses Feast with a **local SQLite backend**.

---

# Objectives

- Initialize a Feast Feature Repository
- Define Entity, Data Source and Feature View
- Materialize features into the online store
- Retrieve historical features from the offline store
- Train an IRIS classification model using Feast
- Retrieve online features for inference
- Demonstrate that predictions from Feast match predictions using raw data

---

# Project Structure

```
example-versioning/
│
├── feature_repo/
│   └── feature_repo/
│       ├── feature_store.yaml
│       ├── feature_definitions.py
│       ├── data/
│       │   ├── iris_features.parquet
│       │   ├── registry.db
│       │   └── online_store.db
│       └── test_workflow.py
│
├── prepare_feast_data.py
├── train_feast.py
├── predict_feast.py
├── iris_model.pkl
├── train.py
└── metrics.csv
```

---

# Feast Architecture

```
                Raw IRIS Dataset
                       │
                       ▼
            prepare_feast_data.py
                       │
                       ▼
            iris_features.parquet
                       │
                       ▼
          Feast Feature Repository
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
 Offline Store                  Online Store
 (Training)                    (Inference)
         │                           │
         ▼                           ▼
 Random Forest Model          Feature Retrieval
         │                           │
         └─────────────┬─────────────┘
                       ▼
                 Model Prediction
```

---

# Feature Store Components

## Entity

```
sample_id
```

Each IRIS sample is uniquely identified using `sample_id`.

---

## Data Source

```
iris_features.parquet
```

The IRIS dataset is stored as a Parquet file with an additional timestamp column required by Feast.

---

## Feature View

The following features are stored inside Feast:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

# Offline Store

Used during model training.

Historical features are retrieved using:

```python
store.get_historical_features(...)
```

---

# Online Store

Used during inference.

Online features are retrieved using:

```python
store.get_online_features(...)
```

---

# Model

Algorithm Used:

```
RandomForestClassifier
```

Training Accuracy:

```
1.0
```

---

# Workflow

1. Prepare Feast-compatible IRIS dataset
2. Initialize Feast Repository
3. Define Entity
4. Define Data Source
5. Define Feature View
6. Apply Feast Definitions
7. Materialize Features
8. Retrieve Offline Features
9. Train Random Forest Model
10. Retrieve Online Features
11. Predict IRIS Class

---

# Commands Used

## Initialize Feast

```bash
feast init feature_repo
```

---

## Register Feature Definitions

```bash
feast apply
```

---

## Materialize Features

```bash
feast materialize 2024-01-01T00:00:00 2025-01-01T00:00:00
```

---

## Train Model

```bash
python train_feast.py
```

---

## Predict

```bash
python predict_feast.py
```

---

# Output

Training:

```
Training Accuracy: 1.0
Model saved as iris_model.pkl
```

Prediction:

```
Prediction using Feast Features:
Setosa

Prediction using Raw Data:
Setosa

Predictions Match:
True
```

---

# Learning Outcomes

- Learned why feature stores are required in production ML systems.
- Understood the difference between Offline Store and Online Store.
- Implemented Feast using a local SQLite backend.
- Trained a model using historical features.
- Performed real-time inference using Feast.
- Verified training-serving consistency.

---

# Technologies Used

- Python
- Feast
- Scikit-Learn
- Pandas
- SQLite
- Parquet
- Git
- GitHub

---

# Author

**Parveen Saroha**

IIT Madras - BS in Data Science and Applications