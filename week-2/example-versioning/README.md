# MLOps Week 2 Assignment – Data Version Control (DVC)

## Student Information

**Name:** Parveen Saroha
**Roll Number:** 21f3001560

---

# Objective

The objective of this assignment is to understand and implement **Data Version Control (DVC)** for Machine Learning projects.

The assignment demonstrates how **Git** and **DVC** work together to version:

* Machine Learning datasets
* Trained model artifacts
* Training metrics

Google Cloud Storage (GCS) is used as the remote storage backend for DVC to enable reproducible machine learning workflows.

---

# Problem Statement

Traditional Git is designed for source code versioning but is not suitable for storing large datasets and trained machine learning models.

DVC extends Git by:

* Tracking datasets and model artifacts
* Storing only lightweight metadata (.dvc files) inside Git
* Storing actual datasets and models inside remote cloud storage
* Allowing reproducible ML experiments
* Switching between different dataset/model versions easily

This assignment demonstrates the complete DVC workflow using Google Cloud Storage.

---

# Technologies Used

* Python 3
* Git
* DVC (Data Version Control)
* Google Cloud Storage (GCS)
* Google Vertex AI Workbench
* TensorFlow / Keras

---

# Project Structure

```
week-2/
└── example-versioning/
    ├── .dvc/
    ├── .env/
    ├── data/
    ├── data.dvc
    ├── model.weights.h5.dvc
    ├── metrics.csv
    ├── train.py
    ├── requirements.txt
    ├── README.md
    └── workflow.txt
```

---

# Google Cloud Storage Remote

DVC Remote Storage:

```
gs://week-2-mlops-21f3001560
```

Configured using:

```bash
dvc remote add -d gcsremote gs://week-2-mlops-21f3001560
```

---

# Assignment Workflow

## Step 1 – Clone Repository

```bash
git clone https://github.com/iterative/example-versioning.git
cd example-versioning
```

---

## Step 2 – Create Virtual Environment

```bash
python3 -m venv .env
source .env/bin/activate
```

---

## Step 3 – Install Dependencies

```bash
pip install -r requirements.txt
pip install "dvc[gs]"
```

---

## Step 4 – Configure DVC Remote

```bash
dvc remote add -d gcsremote gs://week-2-mlops-21f3001560
```

---

# First Iteration (1000 Images)

## Download Dataset

```bash
dvc get https://github.com/iterative/dataset-registry tutorials/versioning/data.zip

unzip -q data.zip
rm -f data.zip
```

---

## Track Dataset

```bash
dvc add data
```

---

## Train Model

```bash
python train.py
```

---

## Track Model

```bash
dvc add model.weights.h5
```

---

## Upload Dataset & Model to GCS

```bash
dvc push
```

---

## Commit First Version

```bash
git add data.dvc model.weights.h5.dvc metrics.csv .gitignore

git commit -m "First iteration with 1000 images"
```

---

## Create Version Tag

```bash
git tag -a v1.0 -m "Model version 1.0 - 1000 images"
```

---

# Second Iteration (2000 Images)

## Download Updated Labels

```bash
dvc get https://github.com/iterative/dataset-registry tutorials/versioning/new-labels.zip

unzip -q new-labels.zip
rm -f new-labels.zip
```

---

## Update Dataset

```bash
dvc add data
```

---

## Retrain Model

```bash
python train.py
```

---

## Track Updated Model

```bash
dvc add model.weights.h5
```

---

## Upload Updated Artifacts

```bash
dvc push
```

---

## Commit Second Version

```bash
git add data.dvc model.weights.h5.dvc metrics.csv

git commit -m "Second iteration with 2000 images"
```

---

## Create Version Tag

```bash
git tag -a v2.0 -m "Model version 2.0 - 2000 images"
```

---

# Switching Between Versions

## Restore Version 1

```bash
git checkout v1.0

dvc pull

dvc checkout
```

---

## Restore Latest Version

```bash
git checkout week_2

dvc pull

dvc checkout
```

---

# DVC Commands Used

```bash
dvc remote add

dvc get

dvc add

dvc push

dvc pull

dvc checkout
```

---

# Git Commands Used

```bash
git clone

git add

git commit

git tag

git checkout

git push
```

---

# Files Tracked by DVC

* data/
* model.weights.h5

Tracked in Git as:

* data.dvc
* model.weights.h5.dvc

---

# Files Tracked by Git

* README.md
* workflow.txt
* train.py
* requirements.txt
* metrics.csv
* data.dvc
* model.weights.h5.dvc
* .dvc/config
* .gitignore

---

# Learning Outcomes

Through this assignment, I learned:

* Git and DVC integration
* Dataset versioning
* Model versioning
* Google Cloud Storage as DVC Remote
* Managing multiple model versions
* Creating reproducible ML pipelines
* Using Git tags for experiment tracking
* Restoring previous dataset and model versions

---

# Conclusion

This assignment successfully demonstrates a complete Machine Learning version control workflow using **Git**, **DVC**, and **Google Cloud Storage**.

Two different versions of the dataset and trained model were created, uploaded to a cloud-based DVC remote, and successfully restored using Git and DVC commands, ensuring a fully reproducible machine learning pipeline.
