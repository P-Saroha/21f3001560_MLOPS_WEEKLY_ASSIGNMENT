# Week 9 — Explainability, Fairness, and Drift in the IRIS Pipeline

## 1. Overview

This assignment extends the IRIS machine learning pipeline with responsible machine learning practices for **fairness, explainability, data drift monitoring, and governance**.

The following components were implemented:

* Sensitive `location` attribute for fairness analysis
* Fairness evaluation using **Fairlearn MetricFrame**
* Model explainability using **SHAP**
* Multiclass SHAP summary plots
* Data drift simulation and detection using the **Kolmogorov-Smirnov (KS) test**
* Distribution comparison plots
* Model governance through a model card

The implementation uses a **Random Forest classifier** trained only on the four original IRIS features.

---

## 2. Learning Objectives

This assignment demonstrates how to:

1. Introduce and audit a sensitive attribute without using it as a model feature.
2. Measure model performance across demographic groups using Fairlearn.
3. Explain multiclass model predictions using SHAP.
4. Interpret SHAP summary plots for the Virginica class.
5. Detect changes in production feature distributions.
6. Distinguish between data drift and concept drift.
7. Document model purpose, performance, limitations, fairness, and monitoring considerations.

---

## 3. Project Structure

```text
week-9/
│
├── data/
│   ├── iris_clean.csv
│   ├── iris_with_location.csv
│   └── iris_production.csv
│
├── plots/
│   ├── shap_setosa.png
│   ├── shap_versicolor.png
│   ├── shap_virginica.png
│   ├── drift_sepal_length.png
│   ├── drift_sepal_width.png
│   ├── drift_petal_length.png
│   └── drift_petal_width.png
│
├── prepare_data.py
├── fairness_analysis.py
├── explainability.py
├── drift_detection.py
├── model_card.md
├── requirements.txt
└── README.md
```

---

# 4. Environment Setup

The experiments were executed in the **Google Cloud Platform (GCP) environment** using a Python virtual environment.

Activate the existing environment:

```bash
source .env/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The main libraries used include:

* Python
* pandas
* NumPy
* scikit-learn
* Fairlearn
* SHAP
* SciPy
* Matplotlib

---

# 5. Task 1 — Introduce the Location Attribute

## Objective

A sensitive `location` attribute was introduced into the IRIS dataset.

The location value was randomly assigned as either:

```text
0
```

or:

```text
1
```

A fixed random seed of `42` was used to make the experiment reproducible.

## Dataset

The original dataset contains:

```text
150 samples
5 columns
```

The resulting dataset contains:

```text
150 samples
6 columns
```

The columns are:

```text
sepal_length
sepal_width
petal_length
petal_width
species
location
```

The generated location distribution was:

```text
location 0 → 71 samples
location 1 → 79 samples
```

## Important Design Decision

The `location` attribute is **not used as a model feature**.

The classifier is trained only using:

```text
sepal_length
sepal_width
petal_length
petal_width
```

The `location` attribute is used only as a sensitive attribute for fairness auditing.

## Generate the Dataset

Run:

```bash
python prepare_data.py
```

This generates:

```text
data/iris_with_location.csv
```

---

# 6. Task 2 — Fairness Assessment with Fairlearn

## Objective

Fairness was evaluated using the **Fairlearn MetricFrame**.

The following metrics were calculated:

* Accuracy
* Precision
* Recall

The metrics were evaluated separately for:

```text
location = 0
location = 1
```

## Model

The classifier used is:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
```

The sensitive attribute was excluded from the training features.

## Feature Configuration

```text
Training Features:
sepal_length
sepal_width
petal_length
petal_width

Sensitive Attribute:
location
```

The implementation verifies that:

```text
location included in training features?
False
```

## Overall Performance

The observed overall performance on the test set was:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.9333 |
| Precision | 0.9333 |
| Recall    | 0.9333 |

## MetricFrame Results

The observed subgroup results were:

| Location | Accuracy | Precision | Recall |
| -------- | -------: | --------: | -----: |
| 0        |   0.8462 |    0.8462 | 0.8462 |
| 1        |   1.0000 |    1.0000 | 1.0000 |

The observed difference between the best and worst groups was:

```text
Accuracy difference  = 0.1538
Precision difference = 0.1538
Recall difference    = 0.1538
```

The test set contained:

```text
Location 0 → 13 samples
Location 1 → 17 samples
```

Because the sensitive attribute was randomly assigned and the subgroup test sets are small, the observed difference should not automatically be interpreted as systematic discrimination. The purpose of the analysis is to demonstrate how subgroup performance can be identified using a fairness audit.

## Run Fairness Analysis

```bash
python fairness_analysis.py
```

The main output is the Fairlearn `MetricFrame`:

```text
          accuracy  precision    recall
location
0         0.846154   0.846154  0.846154
1         1.000000   1.000000  1.000000
```

---

# 7. Task 3 — SHAP Explainability

## Objective

SHAP was used to explain the Random Forest classifier.

A SHAP `TreeExplainer` was applied to the **full IRIS dataset**.

The model uses the four original features:

```text
sepal_length
sepal_width
petal_length
petal_width
```

The sensitive `location` attribute is excluded from the model.

## SHAP Configuration

The SHAP explanation was generated for:

```text
150 samples
4 features
3 classes
```

The resulting SHAP value shape was:

```text
(150, 4, 3)
```

where:

* `150` = number of samples
* `4` = number of model features
* `3` = number of IRIS classes

## Generated Summary Plots

SHAP summary plots were generated for:

```text
Setosa
Versicolor
Virginica
```

The plots are stored in:

```text
plots/shap_setosa.png
plots/shap_versicolor.png
plots/shap_virginica.png
```

## Virginica Feature Importance

The mean absolute SHAP values for the Virginica class were:

| Feature      | Mean Absolute SHAP |
| ------------ | -----------------: |
| petal_length |           0.201497 |
| petal_width  |           0.193650 |
| sepal_length |           0.037120 |
| sepal_width  |           0.007265 |

Therefore, the model's Virginica predictions are influenced most strongly by:

1. `petal_length`
2. `petal_width`

while `sepal_length` and especially `sepal_width` have substantially smaller average contributions.

## Interpreting the Virginica SHAP Plot

The SHAP summary plot can be interpreted as follows:

### SHAP value direction

```text
Negative SHAP  → pushes prediction away from Virginica
Positive SHAP  → pushes prediction toward Virginica
```

Therefore:

```text
Left side  → away from Virginica
Right side → toward Virginica
```

### Feature value color

```text
Red  → high feature value
Blue → low feature value
```

For example, a cluster of red points on the positive side indicates that high values of that feature tend to push the model toward predicting Virginica.

A cluster of blue points on the negative side indicates that low values of that feature tend to push the model away from Virginica.

Each dot represents an individual sample.

SHAP describes how features contribute to the model's prediction. It does not establish that a feature causes the target class.

## Run SHAP Analysis

```bash
python explainability.py
```

---

# 8. Task 4 — Data Drift Detection

## Objective

Data drift was simulated by creating a production dataset whose feature distributions differ from the reference training dataset.

The original:

```text
data/iris_clean.csv
```

was used as the reference dataset.

A simulated production dataset was created by shifting:

```text
petal_length → +1.0
petal_width  → +0.5
```

The following features were left unchanged:

```text
sepal_length
sepal_width
```

The production dataset is:

```text
data/iris_production.csv
```

## Feature Distribution Comparison

The observed means were:

| Feature      | Reference Mean | Production Mean | Mean Shift |
| ------------ | -------------: | --------------: | ---------: |
| sepal_length |         5.8433 |          5.8433 |        0.0 |
| sepal_width  |         3.0573 |          3.0573 |        0.0 |
| petal_length |         3.7580 |          4.7580 |       +1.0 |
| petal_width  |         1.1993 |          1.6993 |       +0.5 |

## Kolmogorov-Smirnov Test

The KS test was used to compare the reference and production distributions.

A significance threshold of:

```text
p < 0.05
```

was used to identify statistically significant distribution differences.

The results were:

| Feature      | KS Statistic |  p-value | Drift |
| ------------ | -----------: | -------: | ----- |
| sepal_length |       0.0000 | 1.000000 | No    |
| sepal_width  |       0.0000 | 1.000000 | No    |
| petal_length |       0.3333 | 8.87e-08 | Yes   |
| petal_width  |       0.3267 | 1.76e-07 | Yes   |

Therefore:

```text
petal_length → DRIFT DETECTED
petal_width  → DRIFT DETECTED
```

while:

```text
sepal_length → NO SIGNIFICANT DRIFT
sepal_width  → NO SIGNIFICANT DRIFT
```

## Distribution Plots

The following plots were generated:

```text
plots/drift_sepal_length.png
plots/drift_sepal_width.png
plots/drift_petal_length.png
plots/drift_petal_width.png
```

## Data Drift vs Concept Drift

### Data Drift

Data drift occurs when the distribution of input features changes:

```text
P(X) changes
```

The experiment in this assignment demonstrates data drift because the distributions of `petal_length` and `petal_width` were shifted in the simulated production data.

### Concept Drift

Concept drift occurs when the relationship between input features and the target changes:

```text
P(Y | X) changes
```

Concept drift cannot be established from the feature-distribution comparison performed in this task. It generally requires monitoring predictions against ground-truth labels over time.

## Important Interpretation

Detecting data drift does not automatically mean that model accuracy has decreased.

Instead, drift is a monitoring signal indicating that production inputs differ from the data used during training. This should trigger further model evaluation and, when appropriate, retraining or other corrective actions.

## Run Drift Detection

```bash
python drift_detection.py
```

---

# 9. Task 5 — Model Governance

A model card is included to document the IRIS classifier and its responsible-ML considerations.

The model card covers:

* Intended use
* Training data
* Model details
* Performance
* Fairness results
* Explainability
* Drift monitoring
* Limitations
* Fairness considerations

The model card is stored as:

```text
model_card.md
```

---

# 10. Responsible ML Summary

The Week-9 pipeline introduces responsible ML practices at multiple stages:

```text
                    IRIS ML Pipeline
                           │
             ┌─────────────┼─────────────┐
             │             │             │
          Training     Evaluation    Monitoring
             │             │             │
             │       ┌─────┴─────┐       │
             │       │           │       │
             │   Fairness      SHAP    Drift
             │   Fairlearn            Detection
             │       │           │       │
             └───────┴───────────┴───────┘
                           │
                      Governance
                           │
                       Model Card
```

These practices help ensure that a production ML system is not evaluated only on overall accuracy but is also monitored for:

* Subgroup performance differences
* Model decision explanations
* Changes in production data
* Model limitations
* Responsible deployment considerations

---

# 11. Reproducing the Assignment

Activate the environment:

```bash
source .env/bin/activate
```

Generate the location-augmented dataset:

```bash
python prepare_data.py
```

Run the fairness analysis:

```bash
python fairness_analysis.py
```

Generate SHAP explanations:

```bash
python explainability.py
```

Run data drift detection:

```bash
python drift_detection.py
```

Check generated outputs:

```bash
find data plots -maxdepth 1 -type f | sort
```

---

# 12. Key Concepts Demonstrated

## Fairness

A model can have high overall accuracy while performing differently across subgroups. Fairlearn's `MetricFrame` makes these subgroup differences visible.

## Explainability

SHAP assigns contribution values to individual features and helps explain why a model makes a particular prediction.

## Data Drift

Data drift occurs when the distribution of production inputs differs from the training distribution.

## Concept Drift

Concept drift occurs when the relationship between features and the target changes.

## Sensitive Attributes

Sensitive attributes can be used for fairness auditing even when they are excluded from model training.

## Governance

Model cards provide documentation about a model's purpose, data, performance, limitations, fairness considerations, and monitoring requirements.

---

# 13. Limitations

This assignment is an educational demonstration and has several limitations:

1. The `location` attribute is randomly generated and does not represent a real demographic characteristic.
2. The fairness evaluation uses a relatively small test set, so subgroup metrics can vary due to sampling.
3. The simulated production drift is artificially introduced and may not represent real-world drift.
4. Data drift alone does not establish concept drift or reduced model accuracy.
5. The model card and fairness analysis are intended as a demonstration of responsible ML practices rather than a complete production governance framework.
6. SHAP explanations describe model behavior and should not be interpreted as causal relationships.

---

# 14. Conclusion

This assignment extends the IRIS machine learning pipeline beyond model training and accuracy evaluation by introducing responsible ML practices.

The final pipeline demonstrates:

```text
Sensitive Attribute
        ↓
Fairness Audit
        ↓
Fairlearn MetricFrame
        ↓
Model Predictions
        ↓
SHAP Explainability
        ↓
Production Monitoring
        ↓
Data Drift Detection
        ↓
Model Governance
```

The implementation demonstrates how fairness, explainability, drift monitoring, and governance can be incorporated into a machine learning workflow to make deployed ML systems more transparent, monitorable, and trustworthy.
