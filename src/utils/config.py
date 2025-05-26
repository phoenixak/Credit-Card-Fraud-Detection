"""
Configuration settings for the Credit Card Fraud Detection project.

This module contains all the configuration parameters used throughout the project,
making it easier to modify settings in one central location.
"""

import os
from pathlib import Path

# Project directory structure
PROJECT_ROOT = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATA_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data settings
DATA_FILE = DATA_DIR / "creditcard.csv"

# Model training settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature settings
FEATURES_TO_SCALE = ["Amount", "Time"]
TARGET_COLUMN = "Class"

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = PROJECT_ROOT / "logs" / "fraud_detection.log"

# Create log directory if it doesn't exist
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Visualization settings
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIGSIZE = (12, 8)
DPI = 100

# Imbalanced data handling
BALANCE_METHOD = "SMOTE"  # Options: "SMOTE", "ADASYN", "RandomOverSampler", "RandomUnderSampler", None

# Model hyperparameters
MODEL_PARAMS = {
    "RandomForestClassifier": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
    },
    "LogisticRegression": {
        "penalty": "l2",
        "C": 1.0,
        "solver": "liblinear",
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
    },
    "XGBClassifier": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "scale_pos_weight": 30,  # Approximate ratio of negative to positive samples
    },
    "LGBMClassifier": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
    },
}
