# Week 8 - MLSecOps Integration into IRIS Pipeline

## Objective

This assignment explores security threats in machine learning systems and demonstrates a data poisoning attack on the IRIS dataset.

The experiment creates poisoned training datasets at 5%, 10%, and 50% corruption levels and measures the impact on model performance using MLflow.

## ML Security Threat Vectors

The assignment covers:

- Data poisoning
- Adversarial examples
- Model extraction
- Prompt injection

## Dataset

The clean IRIS dataset contains:

- 150 samples
- 4 features
- 3 classes

Features:

- sepal_length
- sepal_width
- petal_length
- petal_width

Classes:

- setosa
- versicolor
- virginica

## Data Poisoning

Three poisoned variants were generated:

| Dataset | Poisoned Samples |
|---|---:|
| Clean | 0 |
| 5% | 8 |
| 10% | 15 |
| 50% | 75 |

For each poisoned sample, all four features were replaced with randomly generated values and the class label was randomly assigned.

A fixed random seed of 42 was used for reproducibility.

## ML Model

A Random Forest classifier was trained using:

- n_estimators = 100
- max_depth = 5
- random_state = 42

The same clean test set was used for all experiments so that the effect of poisoning the training data could be compared fairly.

## MLflow Experiment

Experiment name:

`iris_mlsecops_poisoning`

Four runs were logged:

- Clean_0%
- Poisoned_5%
- Poisoned_10%
- Poisoned_50%

The following metrics were logged:

- Accuracy
- Precision
- Recall
- F1 Score

The poisoning level was logged as an MLflow parameter.

## Results

| Poisoning | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0% | 93.33% | 93.33% | 93.33% | 93.33% |
| 5% | 93.33% | 93.33% | 93.33% | 93.33% |
| 10% | 93.33% | 93.33% | 93.33% | 93.33% |
| 50% | 90.00% | 90.24% | 90.00% | 89.97% |

## Analysis

The model did not show measurable degradation at 5% or 10% random poisoning.

Noticeable degradation appeared at 50% poisoning.

Even at 50% corruption, the model achieved 90% accuracy and therefore did not become equivalent to random guessing. This indicates that the remaining clean samples still contained meaningful patterns that the Random Forest could learn.

## Mitigation Strategies

Potential production defenses include:

- Schema validation
- Range and statistical checks
- Anomaly detection
- Class distribution monitoring
- Data provenance tracking
- Dataset versioning
- Access control
- Data quality gates
- Monitoring for unexpected distribution changes

## Data Quantity vs Data Quality

Increasing the amount of data does not automatically solve a poisoning problem. If newly collected data is also contaminated, the amount of poisoned data increases as well.

The clean-data ratio is therefore important. If the clean ratio decreases, more total data may be required to obtain enough reliable clean samples. However, detecting and removing contaminated data should happen before simply collecting more data.

## Files

- `poison_data.py` - generates poisoned datasets
- `train_mlsecops.py` - trains models and logs MLflow experiments
- `data/` - clean and poisoned datasets
- `requirements.txt` - Python dependencies