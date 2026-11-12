"""Evaluate wine cultivar classification with and without dimensionality reduction.

Compares logistic regression on all 13 features, on 2 PCA components and on
2 LDA components (10-fold cross-validation + held-out test set), and saves a
side-by-side plot of both projections.

Usage:
    python analysis.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
TARGET = "Customer_Segment"
COLORS = ["#E4572E", "#17BEBB", "#76B041"]


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "wine.csv")
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    setups = {
        "All 13 features": [],
        "PCA (2 components)": [PCA(n_components=2)],
        "LDA (2 components)": [LinearDiscriminantAnalysis(n_components=2)],
    }
    rows = []
    for name, steps in setups.items():
        pipe = make_pipeline(StandardScaler(), *steps, LogisticRegression(max_iter=1000))
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv)
        test = pipe.fit(X_train, y_train).score(X_test, y_test)
        rows.append(
            {
                "setup": name,
                "cv_accuracy": f"{cv_scores.mean():.1%} ± {cv_scores.std():.1%}",
                "test_accuracy": f"{test:.1%}",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, reducer, labels) in zip(
        axes,
        [
            ("PCA (unsupervised)", PCA(n_components=2), ("PC1", "PC2")),
            ("LDA (supervised)", LinearDiscriminantAnalysis(n_components=2), ("LD1", "LD2")),
        ],
    ):
        Z = make_pipeline(StandardScaler(), reducer).fit_transform(X, y)
        for color, cls in zip(COLORS, sorted(y.unique())):
            mask = (y == cls).to_numpy()
            ax.scatter(Z[mask, 0], Z[mask, 1], c=color, label=f"Cultivar {cls}",
                       edgecolor="white", linewidth=0.5, s=45)
        ratio = reducer.explained_variance_ratio_
        ax.set_title(name)
        ax.set_xlabel(f"{labels[0]} ({ratio[0]:.0%})")
        ax.set_ylabel(f"{labels[1]} ({ratio[1]:.0%})")
        ax.legend()
    fig.tight_layout()
    (ROOT / "reports").mkdir(exist_ok=True)
    fig.savefig(ROOT / "reports" / "pca_vs_lda.png", dpi=110)


if __name__ == "__main__":
    main()
