"""
Main module for the Credit Card Fraud Detection project.

This module provides the main functionality to train and evaluate fraud detection models.
"""

import argparse
import time
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import setup_logger
from src.utils.config import (
    DATA_FILE,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    MODELS_DIR,
    RESULTS_DIR,
)
from src.data.data_loader import (
    load_data,
    preprocess_data,
    scale_features,
    split_features_target,
    balance_classes,
    get_data_stats,
)
from src.models.model_selection import (
    get_models,
    get_recommended_models_for_fraud_detection,
    get_model_grid_search_params,
)
from src.models.model_training import (
    train_test_split_data,
    train_and_evaluate_models,
    save_model,
    get_best_model,
    save_results,
    tune_model_hyperparameters,
)
from src.visualization.visualization import (
    plot_class_distribution,
    plot_correlation_matrix,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
    generate_summary_report,
)

# Set up logger
logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection")

    # Data options
    parser.add_argument(
        "--data-file", type=str, default=None, help="Path to the CSV data file"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Proportion of data to use for testing",
    )

    # Model options
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Models to train and evaluate",
    )
    parser.add_argument(
        "--recommended-models",
        action="store_true",
        help="Use recommended models for fraud detection",
    )
    parser.add_argument(
        "--optimize-threshold",
        action="store_true",
        help="Optimize decision threshold for each model",
    )
    parser.add_argument(
        "--tune-hyperparams",
        action="store_true",
        help="Tune hyperparameters for each model",
    )

    # Class balancing options
    parser.add_argument(
        "--balance-method",
        type=str,
        default=None,
        help="Method for balancing class distribution",
    )

    # Output options
    parser.add_argument(
        "--save-models", action="store_true", help="Save trained models to disk"
    )
    parser.add_argument(
        "--save-results", action="store_true", help="Save evaluation results to disk"
    )
    parser.add_argument(
        "--generate-report", action="store_true", help="Generate visual summary report"
    )

    # Additional options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def run_fraud_detection(args):
    """
    Run the fraud detection process with the given arguments.

    Args:
        args: Command line arguments.
    """
    start_time = time.time()
    logger.info("Starting Credit Card Fraud Detection")

    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    try:
        # Load data
        df = load_data(args.data_file)

        # Display data stats
        stats = get_data_stats(df)
        logger.info(
            f"Dataset statistics: {stats['n_rows']} rows, {stats['n_columns']} columns"
        )
        if "target_distribution" in stats:
            logger.info(
                f"Target distribution: {stats['target_distribution']['counts']}"
            )
            logger.info(f"Fraud ratio: {stats.get('fraud_ratio', 'N/A')}")

        # Preprocess data
        df_preprocessed = preprocess_data(df)

        # Scale features
        df_scaled = scale_features(df_preprocessed)

        # Split features and target
        X, y = split_features_target(df_scaled)

        # Balance classes if requested
        if args.balance_method:
            X, y = balance_classes(X, y, args.balance_method)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split_data(X, y, args.test_size)

        # Plot class distribution
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plot_class_distribution(y, save_path=RESULTS_DIR / "class_distribution.png")

        # Plot correlation matrix
        plot_correlation_matrix(
            df_scaled, save_path=RESULTS_DIR / "correlation_matrix.png"
        )

    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        return

    # Determine which models to use
    if args.recommended_models:
        model_names = get_recommended_models_for_fraud_detection()
        logger.info(f"Using recommended models: {model_names}")
    elif args.models:
        model_names = args.models
        logger.info(f"Using specified models: {model_names}")
    else:
        model_names = None  # Use all available models
        logger.info("Using all available models")

    # Get models
    models = get_models(model_names)

    # Tune hyperparameters if requested
    if args.tune_hyperparams:
        logger.info("Tuning hyperparameters")

        for model_name, model in list(models.items()):
            try:
                # Get hyperparameter grid
                param_grid = get_model_grid_search_params(model_name)

                # Tune hyperparameters
                tuning_results = tune_model_hyperparameters(
                    model, X_train, y_train, param_grid
                )

                # Create new model with optimal parameters
                models[model_name] = get_models(
                    [model_name], {model_name: tuning_results["best_params"]}
                )[model_name]

                logger.info(f"Tuned {model_name}: {tuning_results['best_params']}")

            except Exception as e:
                logger.error(f"Error tuning {model_name}: {e}")

    # Train and evaluate models
    logger.info("Training and evaluating models")
    results = train_and_evaluate_models(
        models, X_train, y_train, X_test, y_test, args.optimize_threshold
    )

    # Get best model
    try:
        best_model_name, best_model = get_best_model(results)
        logger.info(f"Best model: {best_model_name}")

        # Plot feature importance if available
        try:
            if hasattr(best_model, "feature_importances_"):
                plot_feature_importance(
                    X.columns.tolist(),
                    best_model.feature_importances_,
                    title=f"{best_model_name} - Feature Importance",
                    save_path=RESULTS_DIR / "feature_importance.png",
                    top_n=20,
                )
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")

        # Save best model if requested
        if args.save_models:
            logger.info("Saving models")
            os.makedirs(MODELS_DIR, exist_ok=True)

            # Save best model
            save_model(
                best_model,
                best_model_name,
                metadata={
                    "metrics": results[best_model_name]["evaluation"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset_stats": {
                        "n_rows": stats["n_rows"],
                        "n_columns": stats["n_columns"],
                        "fraud_ratio": stats.get("fraud_ratio", None),
                    },
                    "optimal_threshold": results[best_model_name]["optimal_threshold"],
                },
            )
    except Exception as e:
        logger.error(f"Error processing best model: {e}")

    # Save results if requested
    if args.save_results:
        logger.info("Saving results")
        save_results(results)

    # Generate report if requested
    if args.generate_report:
        logger.info("Generating summary report")
        generate_summary_report(results)

    elapsed_time = time.time() - start_time
    logger.info(f"Completed Credit Card Fraud Detection in {elapsed_time:.2f} seconds")


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()

    # Run fraud detection
    run_fraud_detection(args)


if __name__ == "__main__":
    main()
