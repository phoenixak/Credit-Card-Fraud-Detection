"""
Prediction script for the Credit Card Fraud Detection project.

This script provides functionality to make predictions using a trained model.
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import setup_logger
from src.utils.config import FEATURES_TO_SCALE
from src.data.data_loader import load_data, preprocess_data, scale_features
from src.models.model_training import load_model

# Set up logger
logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection Prediction"
    )

    # Input options
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the trained model file"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to the CSV data file for prediction",
    )

    # Output options
    parser.add_argument(
        "--output-file", type=str, default=None, help="Path to save prediction results"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary classification",
    )

    # Additional options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def predict(model, data: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Make predictions using a trained model.

    Args:
        model: The trained model.
        data (pd.DataFrame): Input data for prediction.
        threshold (float, optional): Decision threshold for binary classification.
            Defaults to 0.5.

    Returns:
        Dict[str, Any]: Dictionary with prediction results.
    """
    logger.info(f"Making predictions with threshold={threshold}")

    try:
        # Make predictions
        y_pred = model.predict(data)

        # Get probability predictions if available
        try:
            y_prob = model.predict_proba(data)[:, 1]  # Probability of positive class
            has_proba = True
        except (AttributeError, IndexError):
            logger.warning("Model does not support probability predictions")
            y_prob = None
            has_proba = False

        # Apply custom threshold if probabilities are available
        if has_proba and threshold != 0.5:
            y_pred = (y_prob >= threshold).astype(int)
            logger.info(f"Applied custom threshold {threshold}")

        # Count predictions
        n_frauds = np.sum(y_pred == 1)
        n_total = len(y_pred)
        fraud_rate = n_frauds / n_total

        logger.info(
            f"Prediction results: {n_frauds} out of {n_total} transactions classified as fraud ({fraud_rate:.2%})"
        )

        # Create results dictionary
        results = {
            "predictions": y_pred.tolist(),
            "n_frauds": int(n_frauds),
            "n_total": int(n_total),
            "fraud_rate": float(fraud_rate),
        }

        # Add probabilities if available
        if has_proba:
            results["probabilities"] = y_prob.tolist()

        return results

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise


def format_prediction_output(
    data: pd.DataFrame, results: Dict[str, Any]
) -> pd.DataFrame:
    """
    Format prediction results for output.

    Args:
        data (pd.DataFrame): Original input data.
        results (Dict[str, Any]): Prediction results.

    Returns:
        pd.DataFrame: DataFrame with input data and prediction results.
    """
    # Create a copy of the input data
    output = data.copy()

    # Add prediction column
    output["fraud_prediction"] = results["predictions"]

    # Add probability column if available
    if "probabilities" in results:
        output["fraud_probability"] = results["probabilities"]

    return output


def save_prediction_results(
    output: pd.DataFrame, results: Dict[str, Any], output_file: Optional[str] = None
) -> None:
    """
    Save prediction results to a file.

    Args:
        output (pd.DataFrame): Formatted prediction output.
        results (Dict[str, Any]): Prediction results.
        output_file (str, optional): Path to save the results.
            If None, results are not saved. Defaults to None.
    """
    if output_file is None:
        logger.info("No output file specified, skipping saving results")
        return

    try:
        # Convert output file to Path object
        output_file = Path(output_file)

        # Create parent directory if it doesn't exist
        if output_file.parent and not output_file.parent.exists():
            output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save prediction results
        output.to_csv(output_file, index=False)

        # Save summary
        summary_file = output_file.with_suffix(".summary.json")
        summary = {
            "n_frauds": results["n_frauds"],
            "n_total": results["n_total"],
            "fraud_rate": results["fraud_rate"],
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)

        logger.info(f"Saved prediction results to {output_file}")
        logger.info(f"Saved prediction summary to {summary_file}")

    except Exception as e:
        logger.error(f"Error saving prediction results: {str(e)}")
        raise


def run_prediction(args):
    """
    Run the prediction process with the given arguments.

    Args:
        args: Command line arguments.
    """
    logger.info("Starting Credit Card Fraud Detection Prediction")

    # Load model
    try:
        logger.info(f"Loading model from {args.model_path}")
        model_result = load_model(args.model_path)

        # Check if model and metadata were returned
        if isinstance(model_result, tuple) and len(model_result) == 2:
            model, metadata = model_result
            logger.info(f"Loaded model with metadata: {metadata}")

            # Use optimal threshold from metadata if available
            if "optimal_threshold" in metadata and args.threshold == 0.5:
                args.threshold = metadata["optimal_threshold"]
                logger.info(f"Using optimal threshold from metadata: {args.threshold}")
        else:
            model = model_result
            logger.info("Loaded model without metadata")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return

    # Load and preprocess data
    try:
        logger.info(f"Loading data from {args.data_file}")
        df = load_data(args.data_file)

        # Check if target column is present and remove it
        if "Class" in df.columns:
            logger.info("Removing target column from input data")
            df = df.drop(columns=["Class"])

        # Preprocess data
        df_preprocessed = preprocess_data(df)

        # Scale features
        df_scaled = scale_features(df_preprocessed)

        logger.info(f"Prepared data with shape: {df_scaled.shape}")

    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        return

    # Make predictions
    try:
        results = predict(model, df_scaled, args.threshold)

        # Format prediction output
        output = format_prediction_output(df, results)

        # Save prediction results
        save_prediction_results(output, results, args.output_file)

        logger.info("Completed prediction process")

    except Exception as e:
        logger.error(f"Error during prediction process: {str(e)}")
        return


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()

    # Run prediction
    run_prediction(args)


if __name__ == "__main__":
    main()
