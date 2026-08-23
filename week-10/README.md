# Week 10 — LLMOps: Gemini Supervised Fine-Tuning and Evaluation

## Overview

This week introduces **LLMOps** concepts into the existing IRIS machine learning pipeline.

The objective is to:

1. Prepare IRIS data for supervised fine-tuning.
2. Create two different dataset representations.
3. Store the datasets in Google Cloud Storage.
4. Convert the datasets into the Gemini-compatible `generateContent` JSONL format.
5. Fine-tune `gemini-2.5-flash-lite` using Vertex AI.
6. Create two tuned model variants.
7. Deploy the tuned models to Vertex AI endpoints.
8. Evaluate both tuned models on the same fixed test dataset.
9. Compare classification accuracy and output-format compliance.
10. Identify the better-performing model.

---

## 1. Project Structure

The Week 10 implementation is organized as follows:

```text
week-10/
│
├── data/
│   ├── v1_train.jsonl
│   ├── v1_validation.jsonl
│   ├── v2_train.jsonl
│   ├── v2_validation.jsonl
│   ├── v1_train_gemini.jsonl
│   ├── v1_validation_gemini.jsonl
│   ├── v2_train_gemini.jsonl
│   ├── v2_validation_gemini.jsonl
│   └── test.csv
│
├── prepare_data.py
├── convert_for_gemini.py
├── evaluate_models.py
│
└── results/
    ├── predictions.csv
    ├── evaluation_results.json
    └── comparison.csv
```

---

## 2. Source Dataset

The source dataset was taken from the clean IRIS dataset created during Week 8:

```
week-8/data/iris_clean.csv
```

The dataset contains:

- 150 total samples
- 4 numerical input features
- 1 target column

**Input Features**
- `sepal_length`
- `sepal_width`
- `petal_length`
- `petal_width`

**Target**
- `species`

**Classes**
- setosa
- versicolor
- virginica

The original dataset is balanced:

| Class      | Count |
|------------|-------|
| setosa     | 50    |
| versicolor | 50    |
| virginica  | 50    |

---

## 3. Fixed Train / Validation / Test Split

A fixed stratified split was used so that both fine-tuned models are trained and evaluated using the same underlying examples.

The final split is:

| Dataset    | Samples |
|------------|---------|
| Training   | 120     |
| Validation | 15      |
| Test       | 15      |
| **Total**  | **150** |

Each split maintains equal class representation.

**Training**

| Class      | Count |
|------------|-------|
| setosa     | 40    |
| versicolor | 40    |
| virginica  | 40    |

**Validation**

| Class      | Count |
|------------|-------|
| setosa     | 5     |
| versicolor | 5     |
| virginica  | 5     |

**Test**

| Class      | Count |
|------------|-------|
| setosa     | 5     |
| versicolor | 5     |
| virginica  | 5     |

This is important because the comparison between V1 and V2 should be based on the same test examples.

---

## 4. Task 1 — Data Preparation

The data preparation script is:

```
week-10/prepare_data.py
```

Run it using:

```bash
python week-10/prepare_data.py
```

The script creates:

- `v1_train.jsonl`
- `v1_validation.jsonl`
- `v2_train.jsonl`
- `v2_validation.jsonl`
- `test.csv`

---

## 5. V1 Dataset Representation

V1 uses a compact structured representation.

Example:

```json
{
  "input_text": "sepal_length: 4.4, sepal_width: 2.9, petal_length: 1.4, petal_width: 0.2",
  "output_text": "setosa"
}
```

The target output is simply the class name:

- `setosa`
- `versicolor`
- `virginica`

**V1 Dataset Size**

| Split      | Records |
|------------|---------|
| Training   | 120     |
| Validation | 15      |

Training class distribution:

| Class      | Count |
|------------|-------|
| setosa     | 40    |
| versicolor | 40    |
| virginica  | 40    |

Validation class distribution:

| Class      | Count |
|------------|-------|
| setosa     | 5     |
| versicolor | 5     |
| virginica  | 5     |

---

## 6. V2 Dataset Representation

V2 uses a natural-language description of the same numerical features.

Example:

```json
{
  "input_text": "A flower specimen has a sepal length of 4.4 cm, sepal width of 2.9 cm, petal length of 1.4 cm, and petal width of 0.2 cm. Identify the iris species.",
  "output_text": "This is Iris setosa."
}
```

The target output uses a natural-language response:

- `This is Iris setosa.`
- `This is Iris versicolor.`
- `This is Iris virginica.`

**V2 Dataset Size**

| Split      | Records |
|------------|---------|
| Training   | 120     |
| Validation | 15      |

Training class distribution:

| Class                     | Count |
|---------------------------|-------|
| This is Iris setosa.      | 40    |
| This is Iris versicolor.  | 40    |
| This is Iris virginica.   | 40    |

Validation class distribution:

| Class                     | Count |
|---------------------------|-------|
| This is Iris setosa.      | 5     |
| This is Iris versicolor.  | 5     |
| This is Iris virginica.   | 5     |

---

## 7. V1 and V2 Use the Same Split

V1 and V2 represent the same underlying examples. The difference is only the representation of the input and expected output.

Therefore:

- V1 training examples = 120
- V2 training examples = 120
- V1 validation examples = 15
- V2 validation examples = 15
- Test examples = 15

This allows a fair comparison between the two fine-tuned models.

---

## 8. Test Dataset

The final test dataset is:

```
week-10/data/test.csv
```

Shape: `(15, 5)`

Class distribution:

| Class      | Count |
|------------|-------|
| setosa     | 5     |
| versicolor | 5     |
| virginica  | 5     |

The test dataset is kept separate from the fine-tuning training and validation datasets.

---

## 9. Google Cloud Storage

A dedicated Google Cloud Storage bucket was created for Week 10:

```
gs://21f3001560week10/
```

The Week 10 datasets were stored under:

```
gs://21f3001560week10/week-10/
```

The initial datasets were uploaded using:

```bash
gcloud storage cp week-10/data/v1_train.jsonl \
    gs://21f3001560week10/week-10/

gcloud storage cp week-10/data/v1_validation.jsonl \
    gs://21f3001560week10/week-10/

gcloud storage cp week-10/data/v2_train.jsonl \
    gs://21f3001560week10/week-10/

gcloud storage cp week-10/data/v2_validation.jsonl \
    gs://21f3001560week10/week-10/
```

---

## 10. Gemini-Compatible Dataset Conversion

During Vertex AI tuning, the original dataset representation produced a compatibility problem:

```
Converting from 'VertexTextBison' to 'GenerateContent'
dataset format is currently not supported for this model.
```

Therefore, Gemini-compatible JSONL files were generated.

The conversion script is:

```
week-10/convert_for_gemini.py
```

Run:

```bash
python week-10/convert_for_gemini.py
```

It generated:

- `v1_train_gemini.jsonl`
- `v1_validation_gemini.jsonl`
- `v2_train_gemini.jsonl`
- `v2_validation_gemini.jsonl`

Each generated file was validated successfully.

---

## 11. Gemini JSONL Files

The final Gemini-compatible files contain:

**V1**

| File                          | Records |
|--------------------------------|---------|
| `v1_train_gemini.jsonl`        | 120     |
| `v1_validation_gemini.jsonl`   | 15      |

**V2**

| File                          | Records |
|--------------------------------|---------|
| `v2_train_gemini.jsonl`        | 120     |
| `v2_validation_gemini.jsonl`   | 15      |

All files were successfully validated.

---

## 12. Upload Gemini-Compatible Files

The final Gemini-compatible datasets were uploaded using:

```bash
gcloud storage cp week-10/data/v1_train_gemini.jsonl \
    gs://21f3001560week10/week-10/

gcloud storage cp week-10/data/v1_validation_gemini.jsonl \
    gs://21f3001560week10/week-10/

gcloud storage cp week-10/data/v2_train_gemini.jsonl \
    gs://21f3001560week10/week-10/

gcloud storage cp week-10/data/v2_validation_gemini.jsonl \
    gs://21f3001560week10/week-10/
```

Final GCS contents:

```
gs://21f3001560week10/week-10/
│
├── v1_train.jsonl
├── v1_validation.jsonl
├── v2_train.jsonl
├── v2_validation.jsonl
│
├── v1_train_gemini.jsonl
├── v1_validation_gemini.jsonl
├── v2_train_gemini.jsonl
└── v2_validation_gemini.jsonl
```

---

## 13. Vertex AI Configuration

Vertex AI was used for supervised fine-tuning.

| Setting        | Value                                 |
|----------------|----------------------------------------|
| Project        | `project-a74bda7f-3a9e-4ff2-a23`      |
| Region         | `us-central1`                         |
| Base Model     | `gemini-2.5-flash-lite`               |
| Tuning Method  | Supervised fine-tuning                |

The task is a classification task with labelled examples, so supervised fine-tuning was selected.

---

## 14. Vertex AI Tuning Configuration

The tuning configuration used:

| Parameter                  | Value            |
|-----------------------------|------------------|
| Number of epochs            | 3                |
| Learning rate multiplier    | 1                |
| Adapter size                 | 4                |
| Intermediate checkpoints    | Enabled          |
| Encryption                  | Google-managed   |

A validation dataset was provided for both tuning jobs.

---

## 15. V1 Fine-Tuning Job

The first model was fine-tuned using `v1_train_gemini.jsonl`, with validation dataset `v1_validation_gemini.jsonl`.

| Property        | Value                                                                 |
|------------------|------------------------------------------------------------------------|
| Tuning Job       | `1758081855447367680`                                                 |
| V1 Tuned Model   | `projects/879328269579/locations/us-central1/models/1024551872796557312@1` |
| Status           | Succeeded                                                              |
| Base Model       | `gemini-2.5-flash-lite`                                               |
| Epochs           | 3                                                                       |
| Adapter Size     | 4                                                                       |

---

## 16. V2 Fine-Tuning Job

The second model was fine-tuned using `v2_train_gemini.jsonl`, with validation dataset `v2_validation_gemini.jsonl`.

| Property        | Value                                                                 |
|------------------|------------------------------------------------------------------------|
| Tuning Job       | `153111538243207168`                                                  |
| V2 Tuned Model   | `projects/879328269579/locations/us-central1/models/6563979414462267392@1` |
| Status           | Succeeded                                                              |
| Base Model       | `gemini-2.5-flash-lite`                                               |
| Epochs           | 3                                                                       |
| Adapter Size     | 4                                                                       |

---

## 17. Vertex AI Endpoints

The tuned models were deployed to Vertex AI endpoints.

Multiple endpoints were created because intermediate checkpoints were also deployed. The final/default checkpoint used for evaluation was **checkpoint 3**.

---

## 18. V1 Evaluation Endpoint

| Property      | Value                                                                            |
|----------------|-------------------------------------------------------------------------------|
| Endpoint       | `projects/879328269579/locations/us-central1/endpoints/3395965357444300800`    |
| Display name   | `iris-week10-v1-raw-1`                                                          |
| Model          | `projects/879328269579/locations/us-central1/models/1024551872796557312`       |
| Checkpoint     | 3                                                                                |

---

## 19. V2 Evaluation Endpoint

| Property      | Value                                                                            |
|----------------|-------------------------------------------------------------------------------|
| Endpoint       | `projects/879328269579/locations/us-central1/endpoints/5046534620875587584`    |
| Display name   | `iris-week10-v2-description`                                                    |
| Model          | `projects/879328269579/locations/us-central1/models/6563979414462267392`       |
| Checkpoint     | 3                                                                                |

---

## 20. Endpoint Testing

The tuned endpoints were tested using the Google GenAI client.

Example V1 request:

```python
from google import genai
from google.genai.types import HttpOptions

client = genai.Client(
    vertexai=True,
    project="project-a74bda7f-3a9e-4ff2-a23",
    location="us-central1",
    http_options=HttpOptions(api_version="v1"),
)

endpoint = (
    "projects/879328269579/locations/us-central1/"
    "endpoints/3395965357444300800"
)

response = client.models.generate_content(
    model=endpoint,
    contents=(
        "sepal_length: 5.1, sepal_width: 3.5, "
        "petal_length: 1.4, petal_width: 0.2"
    ),
)

print(response.text)
```

The endpoint successfully returned a classification-oriented response. For the example above, the model identified the flower as likely **Iris Setosa**.

---

## 21. Task 4 — Model Evaluation

The evaluation script is:

```
week-10/evaluate_models.py
```

Run:

```bash
python week-10/evaluate_models.py
```

The evaluation uses the same `week-10/data/test.csv` for both models.

Total test samples: **15**

---

## 22. Evaluation Metrics

The evaluation calculates:

- **Accuracy** — Percentage of test examples for which the predicted class matches the expected class.

  ```
  Accuracy = Correct Predictions / Total Predictions
  ```

- **Precision** — How many predictions for a class were actually correct.
- **Recall** — How many actual examples of a class were correctly identified.
- **F1 Score** — Combines precision and recall.
- **Format Compliance Rate** — Whether the model response conforms to the expected classification output format.

---

## 23. V1 Evaluation Results

**Accuracy: 66.67%** (10 / 15 correct predictions)

| Class      | Precision | Recall | F1    | Support |
|------------|-----------|--------|-------|---------|
| Setosa     | 1.00      | 0.60   | 0.75  | 5       |
| Versicolor | 1.00      | 0.40   | 0.571 | 5       |
| Virginica  | 0.50      | 1.00   | 0.667 | 5       |

---

## 24. V2 Evaluation Results

**Accuracy: 0.00%** under the implemented evaluation/extraction protocol.

| Class      | Precision | Recall | F1   | Support |
|------------|-----------|--------|------|---------|
| Setosa     | 0.00      | 0.00   | 0.00 | 5       |
| Versicolor | 0.00      | 0.00   | 0.00 | 5       |
| Virginica  | 0.00      | 0.00   | 0.00 | 5       |

---

## 25. Final Model Comparison

| Model           | Accuracy | Format Compliance |
|------------------|----------|--------------------|
| V1_RAW           | 66.67%   | 0.00%              |
| V2_DESCRIPTION   | 0.00%    | 0.00%              |

Therefore, **V1_RAW** performed better than **V2_DESCRIPTION** on the fixed 15-example test set.

---

## 26. V1 vs V2 Analysis

### V1_RAW

V1 uses concise structured feature input:

```
sepal_length: ...
sepal_width: ...
petal_length: ...
petal_width: ...
```

The fine-tuned model achieved **66.67% accuracy** and was able to correctly classify most of the test samples. However, its responses were generally explanatory rather than returning only the class label.

- Classification accuracy: Good relative to V2
- Format compliance: Poor

### V2_DESCRIPTION

V2 uses natural-language prompts such as:

```
A flower specimen has a sepal length of ...
Identify the iris species.
```

The expected answer was `This is Iris setosa.` or the equivalent class.

However, on the test set, the evaluator recorded **0% accuracy** and **0% format compliance**. The model frequently produced explanations, uncertainty statements, or classifications that did not match the expected extraction format.

- Classification accuracy under evaluator: Poor
- Format compliance: Poor

---

## 27. Important Evaluation Note

The V2 result of 0% should be interpreted specifically as the result of the implemented evaluation protocol. The model generated natural-language responses, including responses containing species-related reasoning. The evaluator expected a class that could be extracted into one of: `setosa`, `versicolor`, `virginica`.

Therefore, the reported result represents **performance under the defined evaluation and extraction protocol**, rather than a general statement that the underlying model has absolutely no ability to identify iris species.

---

## 28. Generated Result Files

Task 4 generated three result files:

- `week-10/results/predictions.csv` — Individual predictions and model responses.
- `week-10/results/evaluation_results.json` — Detailed evaluation metrics.
- `week-10/results/comparison.csv` — Final model comparison.

Example:

```csv
model,accuracy,format_compliance_rate
V1_RAW,0.6666666666666666,0.0
V2_DESCRIPTION,0.0,0.0
```

---

## 29. IAM / Authentication Issue Encountered

During programmatic endpoint testing, an IAM error was initially encountered:

```
403 PERMISSION_DENIED
Permission: aiplatform.endpoints.predict
```

The same project was using the authenticated Google account `sarohaparveen002@gmail.com`, which had project-level Owner access.

A direct IAM permission test was performed using the Vertex AI REST API. The test confirmed:

```json
{
  "permissions": [
    "aiplatform.endpoints.predict"
  ]
}
```

The permission was therefore available for the tested endpoint. After configuring the GenAI client with the Vertex AI API version:

```python
HttpOptions(api_version="v1")
```

the endpoint request succeeded.

---

## 30. Authentication Verification

The following command was used to verify authentication:

```bash
gcloud auth application-default print-access-token >/dev/null \
    && echo "ADC OK"
```

Output: `ADC OK`

The active account was also verified:

```bash
gcloud config get-value account
```

Output: `sarohaparveen002@gmail.com`

---

## 31. Technologies Used

The Week 10 implementation uses:

- Python
- Pandas
- Scikit-learn
- Google Cloud Storage
- Google Cloud CLI
- Vertex AI
- Gemini / Gemini 2.5 Flash Lite
- Supervised Fine-Tuning
- Vertex AI Endpoints
- Google GenAI SDK
- JSONL
- Google Cloud IAM

---

## 32. Important Commands

**Check Google Cloud project**

```bash
gcloud config get-value project
```

**List buckets**

```bash
gcloud storage ls
```

**List Week 10 files**

```bash
gcloud storage ls gs://21f3001560week10/week-10/
```

**Upload a dataset**

```bash
gcloud storage cp <LOCAL_FILE> \
    gs://21f3001560week10/week-10/
```

**Run data preparation**

```bash
python week-10/prepare_data.py
```

**Convert datasets**

```bash
python week-10/convert_for_gemini.py
```

**Evaluate models**

```bash
python week-10/evaluate_models.py
```

**List Vertex AI endpoints**

```bash
gcloud ai endpoints list \
    --region=us-central1 \
    --project=project-a74bda7f-3a9e-4ff2-a23
```

**Describe an endpoint**

```bash
gcloud ai endpoints describe <ENDPOINT_ID> \
    --region=us-central1 \
    --project=project-a74bda7f-3a9e-4ff2-a23
```

---

## 33. Reproducibility

The main data preparation process can be reproduced using:

```bash
python week-10/prepare_data.py
```

Then convert the datasets:

```bash
python week-10/convert_for_gemini.py
```

Upload the Gemini datasets:

```bash
gcloud storage cp week-10/data/v1_train_gemini.jsonl \
    gs://21f3001560week10/week-10/

gcloud storage cp week-10/data/v1_validation_gemini.jsonl \
    gs://21f3001560week10/week-10/

gcloud storage cp week-10/data/v2_train_gemini.jsonl \
    gs://21f3001560week10/week-10/

gcloud storage cp week-10/data/v2_validation_gemini.jsonl \
    gs://21f3001560week10/week-10/
```

After the tuned models are available and deployed, run:

```bash
python week-10/evaluate_models.py
```

---

## 34. Final Outcome

The Week 10 LLMOps pipeline successfully demonstrated:

```
IRIS Dataset
      ↓
Fixed Train / Validation / Test Split
      ↓
V1 and V2 Dataset Representations
      ↓
Gemini-Compatible JSONL Conversion
      ↓
Google Cloud Storage
      ↓
Vertex AI Supervised Fine-Tuning
      ↓
Gemini 2.5 Flash Lite
      ↓
Two Tuned Models
      ↓
Vertex AI Endpoints
      ↓
Common Test Dataset
      ↓
Model Evaluation
      ↓
V1 vs V2 Comparison
```

The final measured results were:

- **V1_RAW** — Accuracy: 66.67%
- **V2_DESCRIPTION** — Accuracy: 0.00%

Under the implemented evaluation protocol, **V1_RAW** was the better-performing model.

---

## 35. Conclusion

This assignment demonstrates an end-to-end LLMOps workflow using Google Cloud Vertex AI.

The main learning outcomes are:

- Preparing labelled datasets for LLM supervised fine-tuning.
- Maintaining identical train/validation/test splits for fair comparison.
- Converting datasets into Gemini-compatible JSONL format.
- Uploading datasets to Google Cloud Storage.
- Fine-tuning Gemini 2.5 Flash Lite using Vertex AI.
- Deploying tuned models through Vertex AI endpoints.
- Performing programmatic model inference.
- Evaluating LLM outputs using classification metrics.
- Measuring output-format compliance.
- Comparing different dataset representations and their effect on model performance.

The experiment showed that dataset representation matters. In this implementation, the structured V1 representation produced substantially better test-set classification accuracy than the natural-language V2 representation.

**Final Results**

| Model           | Test Accuracy | Format Compliance | Result |
|------------------|----------------|---------------------|--------|
| V1_RAW           | 66.67%         | 0.00%               | Best   |
| V2_DESCRIPTION   | 0.00%          | 0.00%               | Lower  |

- **Best-performing model:** V1_RAW
- **Base model:** gemini-2.5-flash-lite
- **Platform:** Google Cloud Vertex AI
- **Region:** us-central1