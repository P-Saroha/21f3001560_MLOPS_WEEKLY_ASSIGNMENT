# Week 1 - MLOps Assignment

## Student Details

* **Name:** Parveen Saroha
* **IITM BS Roll Number:** 21F3001560

---

## Assignment Objective

Build an end-to-end Machine Learning pipeline on Google Cloud Platform using Vertex AI Workbench and Google Cloud Storage (GCS) for the IRIS classification dataset.

---

## Technologies Used

* Google Cloud Platform (GCP)
* Vertex AI Workbench
* Google Cloud Storage (GCS)
* Python
* Pandas
* Scikit-learn
* Joblib
* Google Cloud Storage Python SDK

---

## Project Structure

```text
week-1/
│
├── src/
│   ├── train.py
│   └── inference.py
│
├── v1/
│   └── data.csv
│
├── v2/
│   └── data.csv
│
├── raw/
│   └── iris.csv
│
└── README.md
```

---

## GCS Buckets

### Data Bucket

```text
gs://21f3001560-iris-data
```

Contains:

```text
data/raw/iris.csv
data/v1/data.csv
data/v2/data.csv
```

### Artifact Bucket

```text
gs://21f3001560-iris-artifacts
```

Stores:

```text
<TIMESTAMP>/
├── model.joblib
├── metrics.json
└── predictions.csv
```

---

## Training Pipeline

1. Download dataset from GCS.
2. Split dataset into train and test sets.
3. Train a Decision Tree Classifier.
4. Evaluate model performance.
5. Save model and metrics.
6. Upload artifacts to GCS under a timestamp folder.

Run:

```bash
python src/train.py
```

---

## Inference Pipeline

1. Download latest trained model from GCS.
2. Load evaluation dataset.
3. Run predictions.
4. Save predictions.
5. Upload predictions to GCS.

Run:

```bash
python src/inference.py
```

---

## Model Details

* Algorithm: Decision Tree Classifier
* Max Depth: 3
* Train-Test Split: 70-30
* Random State: 42

---

## Results

Example Training Run:

```text
Accuracy: 0.967741935483871
```

Artifacts generated:

```text
model.joblib
metrics.json
predictions.csv
```

---

## Assignment Tasks Completed

* [x] GCP Setup
* [x] Vertex AI Workbench Setup
* [x] GCS Data Storage
* [x] Training Pipeline
* [x] Artifact Storage in GCS
* [x] Separate Inference Script
* [x] Timestamp-based Artifact Organization
* [x] Multiple Pipeline Executions

---

## Repository

Branch:

```text
week_1
```

GitHub Repository:

```text
https://github.com/P-Saroha/21f3001560_MLOPS_WEEKLY_ASSIGNMENT
```
