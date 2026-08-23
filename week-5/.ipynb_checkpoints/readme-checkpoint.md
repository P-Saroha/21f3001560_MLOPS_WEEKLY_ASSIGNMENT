# Week 5 – Integrating MLflow into the IRIS Pipeline

## Student Information

**Name:** Parveen Saroha

**Roll Number:** 21f3001560

---

# Problem Statement

The objective of this assignment is to integrate **MLflow** into the existing IRIS Machine Learning pipeline developed in the previous assignments.

The pipeline already included:

- Google Vertex AI Workbench
- Feast Feature Store
- Version-controlled dataset
- Random Forest classifier

This assignment extends the pipeline by introducing **MLflow Experiment Tracking** and **MLflow Model Registry** for experiment management, model versioning, and deployment.

---

# Assignment Objectives

- Perform hyperparameter tuning.
- Track all experiments using MLflow.
- Log model parameters and evaluation metrics.
- Register trained models in the MLflow Model Registry.
- Modify the prediction pipeline to load models directly from the MLflow Registry.

---

# Project Structure

```text
week-5/
│
├── feature_repo/
│   ├── feature_store.yaml
│   ├── feature_definitions.py
│   ├── data/
│   └── registry.db
│
├── train_feast.py
├── predict_feast.py
├── requirements.txt
├── label_encoder.pkl
├── mlflow.db
├── mlruns/
└── README.md
```

---

# Pipeline Architecture

```text
              Feast Feature Store
                     │
                     ▼
         Historical Feature Retrieval
                     │
                     ▼
            Train/Test Split
                     │
                     ▼
        Hyperparameter Tuning
                     │
                     ▼
      Random Forest Model Training
                     │
                     ▼
      MLflow Experiment Tracking
                     │
                     ▼
      MLflow Model Registry
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
 Experiment History      Prediction Pipeline
                                 │
                                 ▼
               Load Registered Model
                                 │
                                 ▼
                           Prediction
```

---

# Tasks Completed

## Task 1 – Hyperparameter Tuning

Performed manual hyperparameter tuning using Random Forest.

### Hyperparameters

```python
max_depth = [3, 5, 10]
n_estimators = [50, 100]
```

Total experiments:

```
3 × 2 = 6 Runs
```

Each configuration was trained and evaluated independently.

---

## Task 2 – MLflow Experiment Tracking

Integrated MLflow into the training pipeline.

### Parameters Logged

- max_depth
- n_estimators
- random_state

### Metrics Logged

- Accuracy
- Precision
- Recall
- F1 Score

### Artifact Logged

- Trained Random Forest Model

Each training configuration is stored as a separate MLflow Run.

---

## Task 3 – Experiment Comparison

Compared all experiment runs using the MLflow Dashboard.

The dashboard provides:

- Experiment history
- Parameter comparison
- Metric comparison
- Run artifacts
- Best-performing model

---

## Task 4 – MLflow Model Registry

Registered every trained model into the MLflow Model Registry.

Registered Model:

```
iris_random_forest
```

Model versions:

- Version 1
- Version 2
- Version 3
- Version 4
- Version 5
- Version 6

---

## Task 5 – Prediction from MLflow Registry

The prediction pipeline was updated to remove dependency on locally loaded model files.

### Previous Workflow

```text
Prediction
    │
    ▼
Load iris_model.pkl
    │
    ▼
Prediction
```

### Updated Workflow

```text
Prediction
    │
    ▼
MLflow Model Registry
    │
    ▼
Load Latest Registered Model
    │
    ▼
Prediction
```

The latest registered model is automatically loaded during inference.

---

# MLflow Configuration

Tracking URI

```text
sqlite:///mlflow.db
```

Experiment Name

```text
iris_feast_experiments
```

Registered Model

```text
iris_random_forest
```

---

# Hyperparameter Configurations

| Experiment | max_depth | n_estimators |
|------------|----------:|-------------:|
| 1 | 3 | 50 |
| 2 | 3 | 100 |
| 3 | 5 | 50 |
| 4 | 5 | 100 |
| 5 | 10 | 50 |
| 6 | 10 | 100 |

---

# Commands Used

## Activate Virtual Environment

```bash
source .env/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Model

```bash
python train_feast.py
```

---

## Launch MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Run Prediction

```bash
python predict_feast.py
```

---

## View Registered Models

```bash
python -c "
from mlflow import MlflowClient
client = MlflowClient(tracking_uri='sqlite:///mlflow.db')
for mv in client.search_model_versions(\"name='iris_random_forest'\"):
    print(mv.version)
"
```

---

# Workflow

1. Load historical features from Feast.
2. Prepare the training dataset.
3. Perform train-test split.
4. Execute hyperparameter tuning.
5. Train Random Forest models.
6. Evaluate model performance.
7. Log experiments into MLflow.
8. Register trained models in MLflow.
9. Compare experiment runs.
10. Load latest registered model during prediction.
11. Retrieve online features from Feast.
12. Generate predictions.

---

# Results

Successfully implemented:

- Hyperparameter tuning
- MLflow experiment tracking
- Model parameter logging
- Metric logging
- Model versioning
- MLflow Model Registry
- Registry-based prediction pipeline

The complete IRIS pipeline now supports experiment tracking, model management, and model versioning using MLflow while continuing to use Feast Feature Store for feature management.

---

# Technologies Used

- Python
- Scikit-learn
- Feast Feature Store
- MLflow
- Pandas
- Joblib
- SQLite
- Google Vertex AI Workbench

---

# Conclusion

This assignment successfully integrated MLflow into the existing Feast-based IRIS Machine Learning pipeline.

The system now supports:

- Experiment tracking
- Hyperparameter comparison
- Model versioning
- Centralized model registry
- Registry-based model loading

This creates a more reproducible, scalable, and production-ready MLOps workflow.