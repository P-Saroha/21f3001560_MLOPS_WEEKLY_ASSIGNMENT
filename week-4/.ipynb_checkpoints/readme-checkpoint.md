# Week 4 – Continuous Integration for IRIS ML Pipeline

## Problem Statement

In the previous assignments:

- Week 1: Built an IRIS classification pipeline.
- Week 2: Added DVC for data and model versioning.
- Week 3: Integrated Feast Feature Store for feature management.

Although the pipeline was reproducible, there was no automatic mechanism to verify that future code changes would not break the pipeline.

The objective of Week 4 is to integrate Continuous Integration (CI) using GitHub Actions so that every Push and Pull Request automatically:

- Fetches versioned artifacts using DVC
- Runs data validation tests
- Runs model evaluation tests
- Generates a test report
- Publishes the report using CML
- Prevents broken code from being merged

---

# Repository Structure

```
21f3001560_MLOPS_WEEKLY_ASSIGNMENT/

├── week-2/
│   └── example-versioning/
│       ├── .dvc/
│       ├── data.dvc
│       ├── model.weights.h5.dvc
│       └── train.py
│
├── week-3/
│   ├── feature_repo/
│   ├── train_feast.py
│   ├── predict_feast.py
│   ├── iris_model.pkl
│   └── label_encoder.pkl
│
├── week-4/
│   ├── tests/
│   │   ├── test_data_validation.py
│   │   ├── test_model.py
│   │   └── test_prediction.py
│   │
│   ├── reports/
│   ├── requirements.txt
│   └── README.md
│
└── .github/
    └── workflows/
        └── ci.yml
```

---

# Objectives

- Validate dataset quality automatically.
- Validate trained model automatically.
- Pull versioned artifacts using DVC.
- Run CI on every Push and Pull Request.
- Publish CI reports using CML.

---

# Technologies Used

- Python
- Git
- GitHub
- GitHub Actions
- DVC
- Google Cloud Storage (GCS)
- Feast
- Pytest
- CML
- Scikit-learn
- Pandas
- Joblib

---

# Task 1 – Data Validation Tests

Implemented using Pytest.

Checks performed:

- Dataset exists
- Dataset is not empty
- Required columns exist
- No missing values
- Numeric column validation
- Feature value range validation

---

# Task 2 – Model Evaluation Tests

Implemented tests for:

- Model file exists
- Label encoder exists
- Model loads successfully
- Label encoder loads successfully
- Model exposes predict() method
- Prediction matches expected output

---

# Task 3 – GitHub Actions

Created workflow:

```
.github/workflows/ci.yml
```

Pipeline:

1. Checkout repository
2. Setup Python
3. Install dependencies
4. Authenticate with Google Cloud
5. Pull DVC artifacts
6. Execute Pytest
7. Generate report
8. Publish report using CML

---

# Task 4 – Continuous Integration

Workflow triggers:

```
Push
Pull Request
```

Every commit automatically executes:

- DVC Pull
- Unit Tests
- Report Generation

---

# Task 5 – CML Integration

After successful testing, GitHub Actions posts a report on the Pull Request using CML.

Example report:

```
12 Tests Passed

✔ Data Validation
✔ Model Validation
✔ Prediction Tests
```

---

# Task 6 – Pull Request

Created Pull Request:

week_4 → main

Verified:

- CI passed
- DVC Pull successful
- Test report generated
- CML report published

Finally merged into main.

---

# Test Summary

```
12 Passed

test_data_validation.py
test_model.py
test_prediction.py
```

---

# Outcome

Successfully implemented an automated Continuous Integration pipeline for the IRIS ML project.

The workflow automatically validates code quality, verifies the trained model, retrieves versioned artifacts using DVC, and generates automated reports for every Push and Pull Request.