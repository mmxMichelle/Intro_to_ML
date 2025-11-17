import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.decomposition import PCA

plt.style.use("seaborn-v0_8-whitegrid")

# =============================
# settings
# =============================
DATA_DIMENSIONS = [10, 12, 14, 16, 18, 20]
DATA_PATH = "data"
RESULTS_PATH = "results"
os.makedirs(RESULTS_PATH, exist_ok=True)


# =============================
# define visualization function
# =============================

def plot_feature_distributions(df, y, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col, hue=y, kde=True, element="step", palette="Set1")
        plt.title(f"Feature Distribution: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{col}_distribution.png"), dpi=200)
        plt.close()


def plot_correlation_heatmap(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation_heatmap.png"), dpi=200)
    plt.close()


def plot_pca_scatter(df, y, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df)
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["label"] = y

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="label", palette="Set1", alpha=0.8)
    plt.title(f"PCA (2 Components) — Explained Var: {pca.explained_variance_ratio_.sum():.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_scatter.png"), dpi=200)
    plt.close()


def plot_pairplot(df, y, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    df_copy = df.copy()
    df_copy["label"] = y
    sns.pairplot(df_copy, hue="label", corner=True, palette="Set1", diag_kind="kde")
    plt.suptitle("Pairplot of All Features", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pairplot.png"), dpi=200)
    plt.close()


def plot_2d_feature_pairs(df, y, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    feature_pairs = list(combinations(df.columns, 2))
    for (f1, f2) in feature_pairs:
        plt.figure(figsize=(5, 4))
        sns.scatterplot(x=df[f1], y=df[f2], hue=y, palette="Set1", alpha=0.8)
        plt.title(f"2D Scatter: {f1} vs {f2}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{f1}_vs_{f2}.png"), dpi=150)
        plt.close()


# =============================
# main stream
# =============================
for dim in DATA_DIMENSIONS:
    print(f"\n=== Processing {dim}-D Dataset ===")

    X_path = os.path.join(DATA_PATH, f"kryptonite-{dim}-X.npy")
    y_path = os.path.join(DATA_PATH, f"kryptonite-{dim}-y.npy")

    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        print(f"⚠️ Missing files for {dim}-D, skipped.")
        continue

    # load data
    X = np.load(X_path)
    y = np.load(y_path)
    df = pd.DataFrame(X, columns=[f"f{i+1}" for i in range(X.shape[1])])

    # create results folder
    dim_dir = os.path.join(RESULTS_PATH, f"{dim}D")
    os.makedirs(dim_dir, exist_ok=True)

    # separate paths to save results
    dirs = {
        "distributions": os.path.join(dim_dir, "distributions"),
        "correlations": os.path.join(dim_dir, "correlations"),
        "PCA": os.path.join(dim_dir, "PCA"),
        "pairplots": os.path.join(dim_dir, "pairplots"),
        "scatter_pairs": os.path.join(dim_dir, "scatter_pairs")
    }

    # plotting
    print("→ Plotting feature distributions...")
    plot_feature_distributions(df, y, dirs["distributions"])

    print("→ Plotting correlation heatmap...")
    plot_correlation_heatmap(df, dirs["correlations"])

    print("→ Plotting PCA scatter...")
    plot_pca_scatter(df, y, dirs["PCA"])

    print("→ Plotting pairplot...")
    plot_pairplot(df, y, dirs["pairplots"])

    print("→ Plotting all 2D feature pairs (this may take some time)...")
    plot_2d_feature_pairs(df, y, dirs["scatter_pairs"])

    print(f"✅ Finished {dim}-D dataset. Results saved in: {dim_dir}")

print("\nAll analyses completed successfully.")
