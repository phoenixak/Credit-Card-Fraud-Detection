"""
Visualization functionality for the Credit Card Fraud Detection project.

This module provides functions for visualizing data and model results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import os
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec

from src.utils.logger import setup_logger
from src.utils.config import PLOT_STYLE, FIGSIZE, DPI, RESULTS_DIR

# Set up logger
logger = setup_logger(__name__)


def set_plot_style():
    """Set the default plot style."""
    plt.style.use(PLOT_STYLE)
    plt.rcParams["figure.figsize"] = FIGSIZE
    plt.rcParams["figure.dpi"] = DPI


def plot_class_distribution(
    y: np.ndarray,
    title: str = "Class Distribution",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot the distribution of classes.

    Args:
        y (np.ndarray): Target variable.
        title (str, optional): Plot title. Defaults to "Class Distribution".
        save_path (str or Path, optional): Path to save the figure.
            If None, the figure is not saved. Defaults to None.

    Returns:
        plt.Figure: The figure object.
    """
    logger.info("Plotting class distribution")

    try:
        # Set plot style
        set_plot_style()

        # Create figure
        fig, ax = plt.subplots()

        # Count classes
        class_counts = pd.Series(y).value_counts().sort_index()

        # Plot class distribution
        sns.countplot(x=y, ax=ax)

        # Annotate bars with count and percentage
        total = len(y)
        for i, count in enumerate(class_counts):
            percentage = count / total * 100
            ax.annotate(
                f"{count} ({percentage:.1f}%)",
                xy=(i, count),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        # Set labels
        ax.set_title(title)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")

        # Set x-tick labels
        ax.set_xticklabels(["Normal (0)", "Fraud (1)"])

        # Save figure if requested
        if save_path:
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"Saved class distribution plot to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Error plotting class distribution: {str(e)}")
        raise


def plot_correlation_matrix(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot the correlation matrix of features.

    Args:
        df (pd.DataFrame): DataFrame with features.
        title (str, optional): Plot title. Defaults to "Correlation Matrix".
        save_path (str or Path, optional): Path to save the figure.
            If None, the figure is not saved. Defaults to None.

    Returns:
        plt.Figure: The figure object.
    """
    logger.info("Plotting correlation matrix")

    try:
        # Set plot style
        set_plot_style()

        # Calculate correlation matrix
        corr = df.corr()

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))

        # Plot correlation matrix
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=0.8,
            vmin=-0.8,
            center=0,
            square=True,
            linewidths=0.5,
            annot=False,
            ax=ax,
        )

        # Set title
        ax.set_title(title)

        # Save figure if requested
        if save_path:
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"Saved correlation matrix plot to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Error plotting correlation matrix: {str(e)}")
        raise


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot the confusion matrix.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        title (str, optional): Plot title. Defaults to "Confusion Matrix".
        save_path (str or Path, optional): Path to save the figure.
            If None, the figure is not saved. Defaults to None.

    Returns:
        plt.Figure: The figure object.
    """
    logger.info("Plotting confusion matrix")

    try:
        # Set plot style
        set_plot_style()

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create figure
        fig, ax = plt.subplots()

        # Plot confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"],
            ax=ax,
        )

        # Set labels
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # Save figure if requested
        if save_path:
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"Saved confusion matrix plot to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")
        raise


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot the ROC curve.

    Args:
        y_true (np.ndarray): True labels.
        y_prob (np.ndarray): Predicted probabilities.
        title (str, optional): Plot title. Defaults to "ROC Curve".
        save_path (str or Path, optional): Path to save the figure.
            If None, the figure is not saved. Defaults to None.

    Returns:
        plt.Figure: The figure object.
    """
    logger.info("Plotting ROC curve")

    try:
        # Set plot style
        set_plot_style()

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)

        # Create figure
        fig, ax = plt.subplots()

        # Plot ROC curve
        ax.plot(fpr, tpr, lw=2)

        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")

        # Set labels
        ax.set_title(title)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        # Set limits
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        # Add grid
        ax.grid(True, alpha=0.3)

        # Save figure if requested
        if save_path:
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"Saved ROC curve plot to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Error plotting ROC curve: {str(e)}")
        raise


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot the Precision-Recall curve.

    Args:
        y_true (np.ndarray): True labels.
        y_prob (np.ndarray): Predicted probabilities.
        title (str, optional): Plot title. Defaults to "Precision-Recall Curve".
        save_path (str or Path, optional): Path to save the figure.
            If None, the figure is not saved. Defaults to None.

    Returns:
        plt.Figure: The figure object.
    """
    logger.info("Plotting Precision-Recall curve")

    try:
        # Set plot style
        set_plot_style()

        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)

        # Create figure
        fig, ax = plt.subplots()

        # Plot Precision-Recall curve
        ax.plot(recall, precision, lw=2)

        # Set labels
        ax.set_title(title)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

        # Set limits
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        # Add grid
        ax.grid(True, alpha=0.3)

        # Save figure if requested
        if save_path:
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"Saved Precision-Recall curve plot to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Error plotting Precision-Recall curve: {str(e)}")
        raise


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    save_path: Optional[Union[str, Path]] = None,
    top_n: Optional[int] = None,
) -> plt.Figure:
    """
    Plot feature importances.

    Args:
        feature_names (List[str]): Names of features.
        importances (np.ndarray): Importance scores.
        title (str, optional): Plot title. Defaults to "Feature Importance".
        save_path (str or Path, optional): Path to save the figure.
            If None, the figure is not saved. Defaults to None.
        top_n (int, optional): Number of top features to show.
            If None, all features are shown. Defaults to None.

    Returns:
        plt.Figure: The figure object.
    """
    logger.info(f"Plotting feature importance with top_n={top_n}")

    try:
        # Set plot style
        set_plot_style()

        # Create dataframe
        df = pd.DataFrame({"Feature": feature_names, "Importance": importances})

        # Sort by importance
        df = df.sort_values("Importance", ascending=False)

        # Select top N features if requested
        if top_n is not None and top_n < len(df):
            df = df.head(top_n)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * len(df))))

        # Plot feature importances
        sns.barplot(x="Importance", y="Feature", data=df, ax=ax)

        # Set labels
        ax.set_title(title)

        # Save figure if requested
        if save_path:
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"Saved feature importance plot to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")
        raise


def plot_threshold_vs_metrics(
    thresholds: np.ndarray,
    metrics: Dict[str, np.ndarray],
    title: str = "Threshold vs. Metrics",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot metrics vs. decision threshold.

    Args:
        thresholds (np.ndarray): Array of threshold values.
        metrics (Dict[str, np.ndarray]): Dictionary mapping metric names to values.
        title (str, optional): Plot title. Defaults to "Threshold vs. Metrics".
        save_path (str or Path, optional): Path to save the figure.
            If None, the figure is not saved. Defaults to None.

    Returns:
        plt.Figure: The figure object.
    """
    logger.info("Plotting threshold vs metrics")

    try:
        # Set plot style
        set_plot_style()

        # Create figure
        fig, ax = plt.subplots()

        # Plot each metric
        for metric_name, metric_values in metrics.items():
            ax.plot(thresholds, metric_values, lw=2, label=metric_name)

        # Set labels
        ax.set_title(title)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")

        # Set limits
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        # Add legend
        ax.legend()

        # Add grid
        ax.grid(True, alpha=0.3)

        # Save figure if requested
        if save_path:
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"Saved threshold vs metrics plot to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Error plotting threshold vs metrics: {str(e)}")
        raise


def plot_model_comparison(
    model_names: List[str],
    metrics: Dict[str, List[float]],
    title: str = "Model Comparison",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot model comparison based on metrics.

    Args:
        model_names (List[str]): Names of models.
        metrics (Dict[str, List[float]]): Dictionary mapping metric names to lists of values.
        title (str, optional): Plot title. Defaults to "Model Comparison".
        save_path (str or Path, optional): Path to save the figure.
            If None, the figure is not saved. Defaults to None.

    Returns:
        plt.Figure: The figure object.
    """
    logger.info("Plotting model comparison")

    try:
        # Set plot style
        set_plot_style()

        # Create dataframe
        df = pd.DataFrame(metrics, index=model_names)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot model comparison
        df.plot(kind="bar", ax=ax)

        # Set labels
        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")

        # Add legend
        ax.legend(title="Metric")

        # Rotate x-tick labels
        plt.xticks(rotation=45, ha="right")

        # Add grid
        ax.grid(True, alpha=0.3, axis="y")

        # Adjust layout
        fig.tight_layout()

        # Save figure if requested
        if save_path:
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
            logger.info(f"Saved model comparison plot to {save_path}")

        return fig

    except Exception as e:
        logger.error(f"Error plotting model comparison: {str(e)}")
        raise


def generate_summary_report(
    results: Dict[str, Dict[str, Any]], save_dir: Optional[Union[str, Path]] = None
) -> None:
    """
    Generate a visual summary report of model results.

    Args:
        results (Dict[str, Dict[str, Any]]): Dictionary of model results.
        save_dir (str or Path, optional): Directory to save the report.
            If None, uses the default directory from config.
            Defaults to None.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR / "reports"

    # Convert to Path object
    save_dir = Path(save_dir)

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Generating summary report in {save_dir}")

    try:
        # Extract model names and metrics
        model_names = []
        metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1": []}

        for model_name, model_result in results.items():
            if "error" not in model_result and "evaluation" in model_result:
                model_names.append(model_name)
                metrics["Accuracy"].append(model_result["evaluation"]["accuracy"])
                metrics["Precision"].append(model_result["evaluation"]["precision"])
                metrics["Recall"].append(model_result["evaluation"]["recall"])
                metrics["F1"].append(model_result["evaluation"]["f1"])

        # Plot model comparison
        if model_names:
            fig = plot_model_comparison(
                model_names,
                metrics,
                title="Model Performance Comparison",
                save_path=save_dir / "model_comparison.png",
            )
            plt.close(fig)

        # For each model, generate individual reports
        for model_name, model_result in results.items():
            if "error" in model_result or "evaluation" not in model_result:
                continue

            logger.info(f"Generating report for model: {model_name}")

            # Create model-specific directory
            model_dir = save_dir / model_name.replace(" ", "_").lower()
            os.makedirs(model_dir, exist_ok=True)

            # Get evaluation metrics
            eval_metrics = model_result["evaluation"]

            # Plot confusion matrix if available
            if "confusion_matrix" in eval_metrics:
                cm = np.array(eval_metrics["confusion_matrix"])
                y_true = np.repeat([0, 1], cm.sum(axis=1))
                y_pred = np.concatenate(
                    [np.repeat([0, 1], cm[0]), np.repeat([0, 1], cm[1])]
                )
                fig = plot_confusion_matrix(
                    y_true,
                    y_pred,
                    title=f"{model_name} - Confusion Matrix",
                    save_path=model_dir / "confusion_matrix.png",
                )
                plt.close(fig)

            # Plot ROC curve if available
            if "roc_curve" in eval_metrics:
                roc_data = eval_metrics["roc_curve"]
                fig = plt.figure(figsize=(10, 8))
                plt.plot(roc_data["fpr"], roc_data["tpr"], lw=2)
                plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(
                    f"{model_name} - ROC Curve (AUC = {eval_metrics.get('roc_auc', 0):.4f})"
                )
                plt.grid(True, alpha=0.3)
                plt.savefig(model_dir / "roc_curve.png", dpi=DPI, bbox_inches="tight")
                plt.close(fig)

            # Plot PR curve if available
            if "pr_curve" in eval_metrics:
                pr_data = eval_metrics["pr_curve"]
                fig = plt.figure(figsize=(10, 8))
                plt.plot(pr_data["recall"], pr_data["precision"], lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(
                    f"{model_name} - Precision-Recall Curve (AP = {eval_metrics.get('average_precision', 0):.4f})"
                )
                plt.grid(True, alpha=0.3)
                plt.savefig(model_dir / "pr_curve.png", dpi=DPI, bbox_inches="tight")
                plt.close(fig)

        logger.info("Summary report generation complete")

    except Exception as e:
        logger.error(f"Error generating summary report: {str(e)}")
        raise
