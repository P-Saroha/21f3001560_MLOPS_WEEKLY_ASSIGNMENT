import json
import re
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai.types import HttpOptions
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


# ============================================================
# Configuration
# ============================================================

PROJECT_ID = "project-a74bda7f-3a9e-4ff2-a23"
LOCATION = "us-central1"

V1_ENDPOINT = (
    "projects/879328269579/locations/us-central1/"
    "endpoints/3395965357444300800"
)

V2_ENDPOINT = (
    "projects/879328269579/locations/us-central1/"
    "endpoints/5046534620875587584"
)

BASE_DIR = Path(__file__).resolve().parent

TEST_PATH = BASE_DIR / "data" / "test.csv"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPECIES = [
    "setosa",
    "versicolor",
    "virginica",
]


# ============================================================
# Gemini client
# ============================================================

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    http_options=HttpOptions(api_version="v1"),
)


# ============================================================
# Prompt builders
# ============================================================

def make_v1_prompt(row):
    return (
        f"sepal_length: {row['sepal_length']}, "
        f"sepal_width: {row['sepal_width']}, "
        f"petal_length: {row['petal_length']}, "
        f"petal_width: {row['petal_width']}"
    )


def make_v2_prompt(row):
    return (
        f"A flower specimen has a sepal length of "
        f"{row['sepal_length']} cm, sepal width of "
        f"{row['sepal_width']} cm, petal length of "
        f"{row['petal_length']} cm, and petal width of "
        f"{row['petal_width']} cm. Identify the iris species."
    )


# ============================================================
# Strict format compliance
# ============================================================

def check_format_compliance(text):
    """
    Valid only if the COMPLETE response is exactly one of:

        setosa
        versicolor
        virginica

    Case-sensitive and no extra text.
    """

    if not isinstance(text, str):
        return False

    return text in SPECIES


# ============================================================
# Extract predicted class for accuracy
# ============================================================

def extract_species(text):
    """
    Extract a species name from the model response.

    This lets us calculate classification accuracy even when
    the model produces explanatory text.

    Format compliance is calculated separately and strictly.
    """

    if not isinstance(text, str):
        return None

    text_lower = text.lower()

    # Check virginica/versicolor first because they contain
    # distinct full names.
    for species in ["virginica", "versicolor", "setosa"]:
        if re.search(rf"\b{species}\b", text_lower):
            return species

    return None


# ============================================================
# Model inference
# ============================================================

def predict(endpoint, prompt):

    response = client.models.generate_content(
        model=endpoint,
        contents=prompt,
    )

    return response.text.strip()


# ============================================================
# Evaluate one model
# ============================================================

def evaluate_model(model_name, endpoint, df, prompt_builder):

    print("\n" + "=" * 70)
    print(f"Evaluating {model_name}")
    print("=" * 70)

    rows = []

    for index, row in df.iterrows():

        expected = row["species"]
        prompt = prompt_builder(row)

        print(
            f"[{index + 1}/{len(df)}] "
            f"Expected={expected}"
        )

        try:

            response = predict(endpoint, prompt)

            predicted = extract_species(response)

            compliant = check_format_compliance(response)

            print(f"  Predicted: {predicted}")
            print(f"  Compliant: {compliant}")
            print(f"  Response: {response[:150]!r}")

            rows.append(
                {
                    "model": model_name,
                    "index": int(index),
                    "expected": expected,
                    "predicted": predicted,
                    "format_compliant": compliant,
                    "response": response,
                }
            )

        except Exception as exc:

            print(f"  ERROR: {exc}")

            rows.append(
                {
                    "model": model_name,
                    "index": int(index),
                    "expected": expected,
                    "predicted": None,
                    "format_compliant": False,
                    "response": f"ERROR: {exc}",
                }
            )

        # Small delay to avoid unnecessary request bursts.
        time.sleep(1)


    result_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = (
        result_df["predicted"] == result_df["expected"]
    ).mean()


    # --------------------------------------------------------
    # Format compliance
    # --------------------------------------------------------

    format_compliance = (
        result_df["format_compliant"].mean()
    )


    # --------------------------------------------------------
    # Per-class precision / recall
    # --------------------------------------------------------

    y_true = result_df["expected"]

    y_pred = result_df["predicted"].fillna("invalid")

    labels = SPECIES

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )


    metrics = {
        "model": model_name,
        "accuracy": float(accuracy),
        "format_compliance_rate": float(format_compliance),
        "per_class": {},
    }


    for i, species in enumerate(labels):

        metrics["per_class"][species] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }


    return result_df, metrics


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("WEEK 10 - TASK 4")
    print("Gemini Fine-Tuned Model Evaluation")
    print("=" * 70)

    df = pd.read_csv(TEST_PATH)

    print("\nTest dataset:")
    print(TEST_PATH)

    print("\nSamples:", len(df))

    print("\nClass distribution:")
    print(df["species"].value_counts().sort_index())


    # --------------------------------------------------------
    # V1
    # --------------------------------------------------------

    v1_results, v1_metrics = evaluate_model(
        "V1_RAW",
        V1_ENDPOINT,
        df,
        make_v1_prompt,
    )


    # --------------------------------------------------------
    # V2
    # --------------------------------------------------------

    v2_results, v2_metrics = evaluate_model(
        "V2_DESCRIPTION",
        V2_ENDPOINT,
        df,
        make_v2_prompt,
    )


    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions = pd.concat(
        [v1_results, v2_results],
        ignore_index=True,
    )

    predictions_path = RESULTS_DIR / "predictions.csv"

    predictions.to_csv(
        predictions_path,
        index=False,
    )


    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    all_metrics = {
        "v1_raw": v1_metrics,
        "v2_description": v2_metrics,
    }

    metrics_path = RESULTS_DIR / "evaluation_results.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_metrics,
            f,
            indent=2,
        )


    # --------------------------------------------------------
    # Comparison table
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "model": "V1_RAW",
                "accuracy": v1_metrics["accuracy"],
                "format_compliance_rate":
                    v1_metrics["format_compliance_rate"],
            },
            {
                "model": "V2_DESCRIPTION",
                "accuracy": v2_metrics["accuracy"],
                "format_compliance_rate":
                    v2_metrics["format_compliance_rate"],
            },
        ]
    )

    comparison_path = RESULTS_DIR / "comparison.csv"

    comparison.to_csv(
        comparison_path,
        index=False,
    )


    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(index=False)
    )


    print("\n" + "=" * 70)
    print("V1 PER-CLASS METRICS")
    print("=" * 70)

    print(
        json.dumps(
            v1_metrics["per_class"],
            indent=2,
        )
    )


    print("\n" + "=" * 70)
    print("V2 PER-CLASS METRICS")
    print("=" * 70)

    print(
        json.dumps(
            v2_metrics["per_class"],
            indent=2,
        )
    )


    print("\nResults saved:")

    print(predictions_path)
    print(metrics_path)
    print(comparison_path)


if __name__ == "__main__":
    main()
