# Week 3 Assignment – Integrating Feast Feature Store into the IRIS Pipeline

## Overview

This assignment extends the existing MLOps project by integrating the **Feast Feature Store** into an IRIS machine learning pipeline.

In the previous assignment, **DVC** was used to version datasets and model artifacts. While DVC ensures reproducibility of datasets and models, it does not guarantee that identical feature engineering logic is applied during both training and inference.

To solve this problem, **Feast** is introduced as a centralized Feature Store. Feast provides a single source of truth for feature definitions, enabling the same engineered features to be used consistently during model training (offline) and real-time prediction (online), thereby eliminating **training-serving skew**.

This implementation uses Feast with a **local SQLite backend**.

---

# Assignment Objectives

- Initialize a Feast Feature Repository.
- Define an Entity, Data Source, and Feature View.
- Register feature definitions using Feast.
- Materialize features into the Online Store.
- Retrieve historical features from the Offline Store.
- Train an IRIS classification model using Feast.
- Retrieve online features for inference.
- Verify that Feast predictions match predictions generated from the raw dataset.

---

# Project Structure

```
21f3001560_MLOPS_WEEKLY_ASSIGNMENT/
│
└── week-2/
    └── example-versioning/
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
        ├── README.md
        ├── workflow.txt
        ├── train.py                (Week-2 starter file)
        └── metrics.csv
```

---

# Feast Pipeline Architecture

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
      ┌──────────────┴──────────────┐
      ▼                             ▼
 Offline Store                 Online Store
(Historical Features)      (Real-time Features)
      │                             │
      ▼                             ▼
 Model Training             Feature Retrieval
      │                             │
      └──────────────┬──────────────┘
                     ▼
               Model Prediction
```

---

# Feast Components

## Entity

```
sample_id
```

Each IRIS record is uniquely identified using the `sample_id` entity.

---

## Data Source

```
iris_features.parquet
```

The IRIS dataset is converted into Parquet format and augmented with:

- sample_id
- event_timestamp

The `event_timestamp` column is required by Feast for point-in-time feature retrieval.

---

## Feature View

The following engineered features are stored inside Feast:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

# Offline Store

The Offline Store is used during model training.

Historical features are retrieved using:

```python
store.get_historical_features(...)
```

These features are then used to train the Random Forest classifier.

---

# Online Store

The Online Store is used during inference.

Features are retrieved in real time using:

```python
store.get_online_features(...)
```

The retrieved feature vector is passed to the trained model for prediction.

---

# Machine Learning Model

**Algorithm Used**

```
RandomForestClassifier
```

**Training Accuracy**

```
1.0
```

---

# Workflow

1. Load the IRIS dataset.
2. Add `sample_id` and `event_timestamp`.
3. Convert the dataset to Parquet format.
4. Initialize the Feast Feature Repository.
5. Define the Entity, Data Source, and Feature View.
6. Apply feature definitions using Feast.
7. Materialize features into the SQLite Online Store.
8. Retrieve historical features from the Offline Store.
9. Train the Random Forest model.
10. Retrieve online features using `sample_id`.
11. Perform inference using the trained model.
12. Compare predictions obtained from Feast with predictions obtained from the raw dataset.

---

# Commands Used

## Initialize Feast Repository

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

## Prepare Dataset

```bash
python prepare_feast_data.py
```

---

## Train Model

```bash
python train_feast.py
```

---

## Perform Inference

```bash
python predict_feast.py
```

---

# Results

## Model Training

```
Training Accuracy: 1.0
Model saved as iris_model.pkl
```

---

## Model Inference

```
Prediction using Feast Features:
Setosa

Prediction using Raw Data:
Setosa

Predictions Match:
True
```

The identical predictions confirm that Feast serves the same feature values used during training, eliminating training-serving skew.

---

# Learning Outcomes

- Understood the role of Feature Stores in production ML systems.
- Learned the difference between Offline Store and Online Store.
- Created a Feast Feature Repository.
- Defined Entities, Data Sources, and Feature Views.
- Materialized features into the Online Store.
- Retrieved historical features for model training.
- Retrieved online features for inference.
- Trained an IRIS classification model using Feast.
- Verified consistency between training and serving features.

---

# Technologies Used

- Python
- Feast
- Scikit-learn
- Pandas
- SQLite
- Parquet
- Git
- GitHub

---

# Assignment Status

| Task | Status |
|------|--------|
| Initialize Feast Repository | ✅ Completed |
| Define Entity | ✅ Completed |
| Define Data Source | ✅ Completed |
| Define Feature View | ✅ Completed |
| Apply Feature Definitions | ✅ Completed |
| Materialize Features | ✅ Completed |
| Offline Feature Retrieval | ✅ Completed |
| Model Training | ✅ Completed |
| Online Feature Retrieval | ✅ Completed |
| Prediction Consistency Verification | ✅ Completed |
| BigQuery Backend (Optional) | Not Implemented |

---

# Author

**Parveen Saroha**

- B.E. Electronics & Communication Engineering, UIET Panjab University
- B.S. Data Science and Applications, IIT Madras