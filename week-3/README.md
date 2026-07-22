# Week 3 Assignment – Integrating Feast Feature Store into the IRIS Pipeline

## Overview

This assignment extends the MLOps pipeline by integrating the **Feast Feature Store** into an IRIS machine learning workflow.

In the previous assignment, **DVC** was used to version datasets and model artifacts. While DVC ensures reproducibility by tracking data and models, it does not guarantee that the same feature engineering logic is used during both model training and inference.

To solve this problem, **Feast** is introduced as a centralized Feature Store. Feast provides a single source of truth for feature definitions, ensuring that identical features are served during offline training and online inference, thereby eliminating **training-serving skew**.

For this assignment, the **official IIT Madras Week 3 time-aware IRIS dataset** was used. The dataset already contains the required entity and timestamp fields, making it suitable for Feast.

The implementation uses Feast with a **local SQLite backend** for both the registry and online feature store.

---

# Assignment Objectives

- Initialize a Feast Feature Repository.
- Define an Entity, Data Source, and Feature View.
- Register feature definitions using Feast.
- Materialize features into the Online Store.
- Retrieve historical features from the Offline Store.
- Train an IRIS classification model using Feast.
- Retrieve online features for inference.
- Verify that predictions using Feast match the actual labels from the dataset.

---

# Project Structure

```
21f3001560_MLOPS_WEEKLY_ASSIGNMENT/
│
└── week-3/
    │
    ├── README.md
    ├── workflow.txt
    ├── requirements.txt
    ├── prepare_feast_data.py
    ├── train_feast.py
    ├── predict_feast.py
    ├── iris_model.pkl
    ├── label_encoder.pkl
    │
    └── feature_repo/
        ├── README.md
        ├── __init__.py
        │
        └── feature_repo/
            ├── feature_store.yaml
            ├── feature_definitions.py
            ├── test_workflow.py
            │
            └── data/
                ├── iris_data_adapted_for_feast.csv
                ├── iris_data_adapted_for_feast.parquet
                ├── registry.db
                └── online_store.db
```

---

# Feast Pipeline Architecture

```
               IITM Time-Aware IRIS Dataset
                          │
                          ▼
              prepare_feast_data.py
            (Dataset Validation Script)
                          │
                          ▼
       iris_data_adapted_for_feast.parquet
                          │
                          ▼
              Feast Feature Repository
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      Offline Store              Online Store
 (Historical Features)      (Real-Time Features)
             │                         │
             ▼                         ▼
      Model Training          Online Feature Retrieval
             │                         │
             └────────────┬────────────┘
                          ▼
                   Model Prediction
```

---

# Feast Components

## Entity

```
iris_id
```

Each iris plant is uniquely identified using the **iris_id** entity.

---

## Data Source

```
iris_data_adapted_for_feast.parquet
```

The official IITM dataset contains:

- iris_id
- event_timestamp
- created_timestamp
- sepal_length
- sepal_width
- petal_length
- petal_width
- species

The original CSV dataset was converted into **Parquet** format because the installed Feast version uses Parquet for file-based feature sources.

---

## Feature View

The following features are registered inside Feast:

- sepal_length
- sepal_width
- petal_length
- petal_width

The **species** column is used as the prediction target (label) during model training.

---

# Offline Store

The Offline Store is used during model training.

Historical feature values are retrieved using:

```python
store.get_historical_features(...)
```

These historical features are then used to train a **Random Forest Classifier**.

---

# Online Store

The Online Store is used during inference.

Features are retrieved in real time using:

```python
store.get_online_features(...)
```

The retrieved feature vector is passed directly to the trained model for prediction.

---

# Machine Learning Model

**Algorithm**

```
RandomForestClassifier
```

**Label Encoding**

The target column (**species**) is encoded using **LabelEncoder** before training.

Both the trained model and label encoder are saved for inference.

Generated files:

- iris_model.pkl
- label_encoder.pkl

---

# Workflow

1. Validate the official IITM IRIS dataset.
2. Initialize the Feast Feature Repository.
3. Define the Entity (iris_id).
4. Define the File Data Source.
5. Define the Feature View.
6. Register feature definitions using Feast.
7. Materialize features into the SQLite Online Store.
8. Retrieve historical features from the Offline Store.
9. Train a Random Forest classifier.
10. Retrieve online features using iris_id.
11. Perform inference using Feast.
12. Compare the predicted species with the actual species.

---

# Commands Used

## Validate Dataset

```bash
python prepare_feast_data.py
```

---

## Register Feature Definitions

```bash
cd feature_repo/feature_repo

feast apply
```

---

## Materialize Features

```bash
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

---

## Train Model

```bash
cd ../..

python train_feast.py
```

---

## Run Inference

```bash
python predict_feast.py
```

---

# Results

## Model Training

```
Historical Features Retrieved From Feast

Training Accuracy: 1.0

Model saved as iris_model.pkl

Label Encoder saved as label_encoder.pkl
```

---

## Model Inference

```
Retrieved Features From Feast

Prediction using Feast Features

versicolor

Actual Species from Dataset

versicolor

Prediction Match: True
```

The matching prediction confirms that Feast serves the same feature values during inference that were used during training, eliminating training-serving skew.

---

# Learning Outcomes

- Understood the role of Feature Stores in production ML systems.
- Learned the difference between Offline Store and Online Store.
- Implemented a Feast Feature Repository.
- Defined Entity, Data Source, and Feature View.
- Registered feature definitions using Feast.
- Materialized features into the Online Store.
- Retrieved historical features for model training.
- Retrieved online features for inference.
- Trained a Random Forest classifier using Feast.
- Used LabelEncoder for categorical labels.
- Verified prediction consistency between Feast and the original dataset.

---

# Technologies Used

- Python
- Feast
- Scikit-learn
- Pandas
- PyArrow
- SQLite
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
| Register Feature Definitions | ✅ Completed |
| Materialize Features | ✅ Completed |
| Offline Feature Retrieval | ✅ Completed |
| Model Training | ✅ Completed |
| Online Feature Retrieval | ✅ Completed |
| Prediction Verification | ✅ Completed |
| BigQuery Backend (Optional) | Not Implemented |

---

# Author

**Parveen Saroha**

- Roll Number: **21F3001560**