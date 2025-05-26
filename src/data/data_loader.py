"""
Data loading functionality for the Credit Card Fraud Detection project.

This module provides functions to load and preprocess the credit card transaction dataset.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from src.utils.config import DATA_FILE, FEATURES_TO_SCALE, TARGET_COLUMN, BALANCE_METHOD
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)


def load_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the credit card transaction dataset from a CSV file.

    Args:
        filepath (str or Path, optional): Path to the CSV file.
            If None, uses the default path from config.
            Defaults to None.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the data file is not found.
    """
    if filepath is None:
        filepath = DATA_FILE

    # Convert to Path object
    filepath = Path(filepath)

    logger.info(f"Loading data from {filepath}")

    try:
        # Check if file exists
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found at path: {filepath}")

        # Load the dataset
        df = pd.read_csv(filepath)

        logger.info(
            f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns"
        )
        return df

    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the credit card transaction dataset by cleaning and transforming the data.

    Args:
        df (pd.DataFrame): The raw DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    logger.info("Preprocessing data")

    try:
        # Make a copy to avoid modifying the original
        df_processed = df.copy()

        # Check for missing values
        missing_values = df_processed.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(
                f"Found missing values in the dataset: {missing_values[missing_values > 0]}"
            )

            # Fill missing values or drop rows depending on the extent of missing data
            if (
                missing_values / len(df_processed)
            ).max() < 0.05:  # Less than 5% missing
                logger.info("Filling missing values")
                # For numeric columns, fill with median
                numeric_cols = df_processed.select_dtypes(include=np.number).columns
                for col in numeric_cols:
                    if missing_values[col] > 0:
                        df_processed[col].fillna(
                            df_processed[col].median(), inplace=True
                        )

                # For categorical columns, fill with mode
                categorical_cols = df_processed.select_dtypes(
                    include=["object"]
                ).columns
                for col in categorical_cols:
                    if missing_values[col] > 0:
                        df_processed[col].fillna(
                            df_processed[col].mode()[0], inplace=True
                        )
            else:
                logger.info("Dropping rows with missing values")
                df_processed.dropna(inplace=True)

        # Remove duplicate rows if any
        n_duplicates = df_processed.duplicated().sum()
        if n_duplicates > 0:
            logger.info(f"Removing {n_duplicates} duplicate rows")
            df_processed.drop_duplicates(inplace=True)

        logger.info(f"Preprocessing complete. Final shape: {df_processed.shape}")
        return df_processed

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise


def scale_features(
    df: pd.DataFrame, features_to_scale: Optional[list] = None
) -> pd.DataFrame:
    """
    Scale numeric features to standardize them.

    Args:
        df (pd.DataFrame): The DataFrame to scale.
        features_to_scale (list, optional): List of features to scale.
            If None, uses the default list from config.
            Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with scaled features.
    """
    logger.info("Scaling features")

    try:
        # Make a copy to avoid modifying the original
        df_scaled = df.copy()

        # Determine which features to scale
        if features_to_scale is None:
            features_to_scale = FEATURES_TO_SCALE

        # Filter to only include features that are actually in the DataFrame
        features_to_scale = [f for f in features_to_scale if f in df_scaled.columns]

        if not features_to_scale:
            logger.warning("No features to scale found in the DataFrame")
            return df_scaled

        logger.info(f"Scaling features: {features_to_scale}")

        # Initialize scaler
        scaler = StandardScaler()

        # Scale features
        df_scaled[features_to_scale] = scaler.fit_transform(
            df_scaled[features_to_scale]
        )

        logger.info("Feature scaling complete")
        return df_scaled

    except Exception as e:
        logger.error(f"Error during feature scaling: {str(e)}")
        raise


def load_and_preprocess(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load and preprocess the credit card transaction dataset in one step.

    Args:
        filepath (str or Path, optional): Path to the CSV file.
            If None, uses the default path from config.
            Defaults to None.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df = load_data(filepath)
    df_preprocessed = preprocess_data(df)
    df_scaled = scale_features(df_preprocessed)
    return df_scaled


def split_features_target(
    df: pd.DataFrame, target_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the DataFrame into features (X) and target (y).

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        target_col (str, optional): Name of the target column.
            If None, uses the default column from config.
            Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: The features and target.
    """
    logger.info("Splitting features and target")

    try:
        # Determine target column
        if target_col is None:
            target_col = TARGET_COLUMN

        # Check if target column is in DataFrame
        if target_col not in df.columns:
            error_msg = f"Target column '{target_col}' not found in DataFrame"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y

    except Exception as e:
        logger.error(f"Error during feature-target split: {str(e)}")
        raise


def balance_classes(
    X: pd.DataFrame, y: pd.Series, method: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balance class distribution using oversampling or undersampling.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        method (str, optional): Method to use for balancing.
            If None, uses the default method from config.
            Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Balanced features and target.
    """
    if method is None:
        method = BALANCE_METHOD

    if method is None:
        logger.info("Skipping class balancing as no method is specified")
        return X, y

    logger.info(f"Balancing class distribution using {method}")

    try:
        # Get class distribution before balancing
        class_counts = y.value_counts()
        logger.info(f"Class distribution before balancing: {class_counts.to_dict()}")

        # Initialize balancing method
        if method.upper() == "SMOTE":
            balancer = SMOTE(random_state=42)
        elif method.upper() == "ADASYN":
            balancer = ADASYN(random_state=42)
        elif method.upper() == "RANDOMOVERSAMPLER":
            balancer = RandomOverSampler(random_state=42)
        elif method.upper() == "RANDOMUNDERSAMPLER":
            balancer = RandomUnderSampler(random_state=42)
        else:
            error_msg = f"Unknown balancing method: {method}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Apply balancing
        X_balanced, y_balanced = balancer.fit_resample(X, y)

        # Convert back to DataFrame/Series to preserve column names
        X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
        y_balanced = pd.Series(y_balanced, name=y.name)

        # Get class distribution after balancing
        balanced_class_counts = y_balanced.value_counts()
        logger.info(
            f"Class distribution after balancing: {balanced_class_counts.to_dict()}"
        )

        return X_balanced, y_balanced

    except Exception as e:
        logger.error(f"Error during class balancing: {str(e)}")
        raise


def get_data_stats(
    df: pd.DataFrame, target_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get statistics about the dataset.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        target_col (str, optional): Name of the target column.
            If None, uses the default column from config.
            Defaults to None.

    Returns:
        Dict[str, Any]: Dictionary with dataset statistics.
    """
    logger.info("Calculating dataset statistics")

    try:
        # Determine target column
        if target_col is None:
            target_col = TARGET_COLUMN

        # Basic dataset info
        stats = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # in MB
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
        }

        # Target distribution if target column is present
        if target_col in df.columns:
            target_counts = df[target_col].value_counts().to_dict()
            target_percentages = (
                df[target_col].value_counts(normalize=True) * 100
            ).to_dict()

            stats["target_distribution"] = {
                "counts": target_counts,
                "percentages": {k: f"{v:.2f}%" for k, v in target_percentages.items()},
            }

            # Fraud ratio
            if 1 in target_counts and 0 in target_counts:
                stats["fraud_ratio"] = target_counts[1] / target_counts[0]

        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            stats["numeric_stats"] = df[numeric_cols].describe().to_dict()

        return stats

    except Exception as e:
        logger.error(f"Error calculating dataset statistics: {str(e)}")
        raise
