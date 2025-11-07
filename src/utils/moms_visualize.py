import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_samples(ax, X_maj, X_min, X_trans, title):
    palette = {
        'majority': "#0171B3",
        'minority': "#DD9908",
        'generated': "#048304"
    }

    # Majority
    ax.scatter(
        X_maj[:, 0], X_maj[:, 1],
        marker='o', s=40,
        color=palette['majority'],
        edgecolor='black', linewidth=0.5,
        alpha=0.6,
        label='Majority'
    )
    # Minority
    ax.scatter(
        X_min[:, 0], X_min[:, 1],
        marker='s', s=40,
        color=palette['minority'],
        edgecolor='black', linewidth=0.5,
        alpha=0.6,
        label='Minority'
    )
    # Generated
    if X_trans.size > 0:
        ax.scatter(
            X_trans[:, 0], X_trans[:, 1],
            marker='x', s=50,
            color=palette['generated'],
            linewidth=1.0,
            alpha=0.8,
            label='Generated'
        )

    ax.set_xlabel('X1', labelpad=6, fontsize=14)
    ax.set_ylabel('X2', labelpad=6, fontsize=14)
    ax.set_title(title, pad=8, fontsize=16)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ax.legend(
    #     frameon=False,
    #     loc='upper right',
    #     fontsize=10
    # )


def plot_tsne(X_org, X_T, y_org, method, seed=1203, save_path=None):
    """
    Generate a t-SNE plot for visualizing the distribution of original and transformed samples.

    Parameters:
    - X_org (ndarray): Original feature vectors (majority + minority).
    - X_T (ndarray): Transformed synthetic samples.
    - y_org (ndarray): Binary labels for X_org (0: majority, 1: minority).
    - method (str): Method name used in the plot title.
    - seed (int): Random seed for t-SNE.
    - save_path (str, optional): If specified, the figure will be saved to this path.
    """
    tsne = TSNE(n_components=2, perplexity=30, early_exaggeration=4, random_state=seed, init="pca")
    
    # Combine data for t-SNE
    X_combined = np.vstack((X_org, X_T))
    y_combined = np.hstack((y_org, np.full(len(X_T), 2)))  # 2: transformed samples
    
    # Compute t-SNE embedding
    X_embedded = tsne.fit_transform(X_combined)

    # Define plot
    plt.figure(figsize=(7, 6))
    plt.rcParams.update({'font.size': 12})
    
    plt.scatter(X_embedded[y_combined == 0, 0], X_embedded[y_combined == 0, 1],
                label="Majority", alpha=0.5, s=40, edgecolor='k', linewidth=0.3)
    
    plt.scatter(X_embedded[y_combined == 1, 0], X_embedded[y_combined == 1, 1],
                label="Minority", alpha=0.5, s=40, edgecolor='k', linewidth=0.3)
    
    plt.scatter(X_embedded[y_combined == 2, 0], X_embedded[y_combined == 2, 1],
                label="Transformed", alpha=0.8, s=50, marker='x', c='red', linewidth=1.0)
    
    plt.title(f"t-SNE of ({method})", fontsize=14)
    plt.xlabel("X1", fontsize=12)
    plt.ylabel("X2", fontsize=12)
    plt.legend(loc="best", fontsize=11, frameon=True)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] t-SNE plot saved to: {save_path}")
    
    plt.show()