"""Inference CLI for saved AgriBot model artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def run_prediction(model_path: str, input_csv: str, output_csv: str | None = None) -> pd.DataFrame:
    """Load model artifact and produce predictions from input CSV."""
    model = joblib.load(model_path)
    data = pd.read_csv(input_csv)
    predictions = model.predict(data)

    result = data.copy()
    result["prediction"] = predictions

    if output_csv:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out, index=False)

    return result


def main() -> None:
    """CLI entrypoint for batch prediction."""
    parser = argparse.ArgumentParser(description="Run crop predictions using saved model pickle.")
    parser.add_argument("--model", required=True, help="Path to agribot_model.pkl")
    parser.add_argument("--input", required=True, help="Path to inference input CSV")
    parser.add_argument("--output", default="artifacts/predictions_output.csv", help="Output CSV path")
    args = parser.parse_args()

    result = run_prediction(args.model, args.input, args.output)
    print(result.head().to_string(index=False))
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
