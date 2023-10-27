import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import structlog
from pathlib import Path
from src.features.calculate_mutual_information import compute_MI

logger = structlog.getLogger(__name__)

def get_stats_sum(data: pd.DataFrame, path: Path) -> None:
    """
    This function calculate the statistical summary of the dataset
    then save the result as a photo of a table.
    param data: a Pandas dataframe
    param path: location to save the photo
    """
    desc = data.describe()
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))  # You can adjust the size as needed
    ax.axis('off')
    # Display the table on the axis
    ax.table(cellText=desc.values,
            colLabels=desc.columns,
            rowLabels=desc.index,
            cellLoc='center',
            loc='center')
    # Save the figure as an image
    fig.tight_layout()
    plt.savefig(Path(path)/'statistical_summary.png')
    logger.info("Save Statistical Summary")

def plot_histograms(data: pd.DataFrame, path: Path) -> None:
    """
    This function plots all histograms of all features in a dataset
    param data: a Pandas dataframe
    param path: location to save the plots
    """
    data.hist(figsize=(15, 10), bins=50)
    plt.suptitle("Histograms of Each Column in Dataset")
    # Save the figure as an image
    plt.savefig(Path(path)/'histograms_of_each_feature.png')
    logger.info("Save Histograms")

def plot_boxplots(data: pd.DataFrame, path: Path, name: str) -> None:
    """
    This function plots all boxplots of all features in a dataset
    param data: a Pandas dataframe
    param path: location to save the plots
    param name: name of a plot to be saved
    """
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=data)
    plt.title("Boxplots of Each Column in Dataset")
    # Save the figure as an image
    plt.savefig(Path(path)/name)
    logger.info("Save Boxplots")

def plot_pairplots(data: pd.DataFrame, path: Path) -> None:
    """
    This function plots pairplots of all features in a dataset
    param data: a Pandas dataframe
    param path: location to save the plots
    """
    plt.figure(figsize=(15, 10))
    sns.pairplot(data=data)
    plt.title("Pairplots of All Features in Dataset")
    # Save the figure as an image
    plt.savefig(Path(path)/"pairplots_of_all_features.png")
    logger.info("Save Pairplots")

def plot_pearson(data: pd.DataFrame, path: Path) -> None:
    """
    This function calculate and make a heatmap of Pearson's Correlation Coefficient
    of all features in a dataset
    param data: a Pandas dataframe
    param path: location to save the plots
    """
    # Pearson's
    correlation_matrix = data.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Pearson's Correlation Coefficient Heatmap")
    # Save the figure as an image
    plt.savefig(Path(path)/"pearson_correlation_heatmap_normalized_data.png")
    logger.info("Save Pearson's Correlation Heatmap")

def plot_spearman(data: pd.DataFrame, path: Path) -> None:
    """
    This function calculate and make a heatmap of Spearman's Rank Correlation
    of all features in a dataset
    param data: a Pandas dataframe
    param path: location to save the plots
    """
    # Spearman's
    spearman_corr = data.corr(method="spearman")
    plt.figure(figsize=(15, 10))
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Spearman's Rank Correlation Heatmap")
    # Save the figure as an image
    plt.savefig(Path(path)/"spearman_rank_correlation_heatmap_normalized_data.png")
    logger.info("Save Spearman's Rank Correlation Heatmap")

def plot_kendall(data: pd.DataFrame, path: Path) -> None:
    """
    This function calculate and make a heatmap of Kendall's Tau Correlation
    of all features in a dataset
    param data: a Pandas dataframe
    param path: location to save the plots
    """
    # Kendall's
    kendall_corr = data.corr(method="kendall")
    plt.figure(figsize=(15, 10))
    sns.heatmap(kendall_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Kendall's Tau Correlation Heatmap")
    # Save the figure as an image
    plt.savefig(Path(path)/"kendall_tau_correlation_heatmap_normalized_data.png")
    logger.info("Save Kendall's Tau Heatmap")

def plot_mutual_information_heatmap(data: pd.DataFrame, path: Path) -> None:
    # Compute Mutual Information matrix
    mi_matrix = pd.DataFrame(index=data.columns, columns=data.columns)
    for col1 in data.columns:
        for col2 in data.columns:
            mi_matrix.loc[col1, col2] = compute_MI(data[col1], data[col2])

    # Convert the MI matrix to float type
    mi_matrix = mi_matrix.astype(float)

    plt.figure(figsize=(15, 10))
    sns.heatmap(mi_matrix, annot=True, cmap='viridis')
    plt.title("Mutual Information Heatmap")
    # Save the figure as an image
    plt.savefig(Path(path)/"mutual_information_heatmap_normalized_data.png")
    logger.info("Save Mutual Information Heatmap")