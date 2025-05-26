# Credit Card Fraud Detection

This project implements a comprehensive machine learning solution for credit card fraud detection. It provides tools for loading and preprocessing transaction data, training multiple fraud detection models, evaluating model performance, and making predictions on new data.

## Features

- **Data Loading & Preprocessing**: Handles missing values, feature scaling, and data cleaning.
- **Class Imbalance Handling**: Implements SMOTE, ADASYN, random oversampling, and undersampling methods.
- **Multiple Models**: Supports 10+ ML algorithms including Random Forest, XGBoost, Logistic Regression, and more.
- **Hyperparameter Tuning**: Grid search optimization for model parameters.
- **Decision Threshold Optimization**: Find optimal classification thresholds to balance precision and recall.
- **Performance Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1, ROC AUC, PR curves.
- **Visualization**: Interactive plots for data exploration and model comparison.
- **Prediction Pipeline**: Ready-to-use prediction system for new data.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project requires the Credit Card Fraud Detection dataset. You can obtain it from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

Place the downloaded `creditcard.csv` file in the `dataset/` directory.

## Usage

### Training Models

```bash
# Train using default settings (all models)
python src/main.py

# Train using recommended models for fraud detection
python src/main.py --recommended-models

# Train with hyperparameter tuning and threshold optimization
python src/main.py --recommended-models --tune-hyperparams --optimize-threshold

# Use SMOTE to handle class imbalance
python src/main.py --balance-method SMOTE

# Save trained models and generate visual reports
python src/main.py --save-models --save-results --generate-report
```

### Making Predictions

```bash
# Make predictions using a trained model
python src/predict.py --model-path models/randomforestclassifier.pkl --data-file dataset/new_transactions.csv --output-file results/predictions.csv

# Use a custom decision threshold
python src/predict.py --model-path models/randomforestclassifier.pkl --data-file dataset/new_transactions.csv --threshold 0.8
```

## Project Structure

```
Credit-Card-Fraud-Detection/
├── dataset/                  # Dataset directory
├── models/                   # Saved models
├── results/                  # Results and reports
├── src/                      # Source code
│   ├── data/                 # Data handling modules
│   │   ├── __init__.py
│   │   └── data_loader.py    # Data loading and preprocessing
│   ├── models/               # Model-related modules
│   │   ├── __init__.py
│   │   ├── model_selection.py # Model configuration
│   │   └── model_training.py  # Training and evaluation
│   ├── utils/                # Utility modules
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration settings
│   │   └── logger.py         # Logging setup
│   ├── visualization/        # Visualization modules
│   │   ├── __init__.py
│   │   └── visualization.py  # Data and results visualization
│   ├── __init__.py
│   ├── main.py               # Main training script
│   └── predict.py            # Prediction script
├── tests/                    # Test modules
├── README.md
└── requirements.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The dataset used in this project is from the [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) Kaggle competition.
- Special thanks to the scikit-learn, XGBoost, and imbalanced-learn communities for their excellent libraries. 