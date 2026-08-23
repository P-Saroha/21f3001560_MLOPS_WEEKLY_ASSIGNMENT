import json
from pathlib import Path


DATA_DIR = Path("week-10/data")


def convert_file(input_file, output_file):
    input_path = DATA_DIR / input_file
    output_path = DATA_DIR / output_file

    count = 0

    with open(input_path, "r", encoding="utf-8") as src, \
         open(output_path, "w", encoding="utf-8") as dst:

        for line in src:
            record = json.loads(line)

            gemini_record = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": record["input_text"]
                            }
                        ]
                    },
                    {
                        "role": "model",
                        "parts": [
                            {
                                "text": record["output_text"]
                            }
                        ]
                    }
                ]
            }

            dst.write(
                json.dumps(gemini_record, ensure_ascii=False)
                + "\n"
            )

            count += 1

    print(f"{output_file}: {count} records")


# V1
convert_file(
    "v1_train.jsonl",
    "v1_train_gemini.jsonl"
)

convert_file(
    "v1_validation.jsonl",
    "v1_validation_gemini.jsonl"
)


# V2
convert_file(
    "v2_train.jsonl",
    "v2_train_gemini.jsonl"
)

convert_file(
    "v2_validation.jsonl",
    "v2_validation_gemini.jsonl"
)


print("\nAll Gemini-compatible datasets created.")
