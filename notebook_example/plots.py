import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression


def plot_correlation_with_target(X, y, target_col="SSPL", save_path=None):
    """
    Grafica la correlacion de cada feature con la columna objetivo.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y are not aligned")

    df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

    correlations = df.corr()[target_col].drop(target_col).sort_values()

    colors = sns.diverging_palette(10, 130, as_cmap=True)
    color_mapped = correlations.map(colors)

    sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

    fig = plt.figure(figsize=(12, 8))
    plt.barh(correlations.index, correlations.values, color=color_mapped)
    plt.title(f"Correlation with {target_col}", fontsize=18)
    plt.xlabel("Correlation Coefficient", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    plt.close(fig)
    return fig


def plot_information_gain_with_target(X, y, target_col="SSPL", save_path=None):
    """
    Grafica la ganancia de informacion (mutual_info_regression) de cada feature con el target.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y are not aligned")

    importances = pd.Series(
        mutual_info_regression(X, y.to_numpy().ravel()), X.columns
    ).sort_values()

    colors = sns.diverging_palette(10, 130, as_cmap=True)
    color_mapped = importances.map(colors)

    sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

    fig = plt.figure(figsize=(12, 8))
    plt.barh(importances.index, importances, color=color_mapped)
    plt.title(f"Information Gain with {target_col}", fontsize=18)
    plt.xlabel("Information Gain", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    plt.close(fig)
    return fig
