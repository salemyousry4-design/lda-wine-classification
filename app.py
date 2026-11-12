"""Streamlit dashboard: wine cultivar classification after LDA dimensionality reduction.

Run with:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

METHOD = "LDA"  # "PCA" or "LDA"
AXES = ("PC1", "PC2") if METHOD == "PCA" else ("LD1", "LD2")
DATA_PATH = Path(__file__).resolve().parent / "data" / "wine.csv"
TARGET = "Customer_Segment"  # the wine cultivar (1, 2 or 3)
COLORS = ["#E4572E", "#17BEBB", "#76B041"]

st.set_page_config(page_title=f"Wine {METHOD} Dashboard", page_icon="🍷", layout="wide")


def reducer():
    if METHOD == "PCA":
        return PCA(n_components=2)
    return LinearDiscriminantAnalysis(n_components=2)


@st.cache_resource
def train():
    df = pd.read_csv(DATA_PATH)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    reduced = Pipeline(
        [
            ("scale", StandardScaler()),
            ("reduce", reducer()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    ).fit(X_train, y_train)
    full = Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
    ).fit(X_train, y_train)
    scores = {
        "reduced": accuracy_score(y_test, reduced.predict(X_test)),
        "full": accuracy_score(y_test, full.predict(X_test)),
    }
    return df, X, y, reduced, scores


df, X, y, model, scores = train()
project = model[:-1]  # scaler + reducer
explained = model.named_steps["reduce"].explained_variance_ratio_

st.title(f"🍷 Wine Cultivar Classification with {METHOD}")
st.write(
    f"13 chemical measurements are compressed into **2 {METHOD} components**, and a "
    "logistic regression classifies each wine into one of three cultivars. Move the "
    "sliders to place your own wine on the map."
)

c1, c2, c3 = st.columns(3)
c1.metric(f"Test accuracy · 2 {METHOD} components", f"{scores['reduced']:.1%}")
c2.metric("Test accuracy · all 13 features", f"{scores['full']:.1%}")
c3.metric(
    "Variance kept by 2 components" if METHOD == "PCA" else "Class separation kept by 2 LDs",
    f"{explained.sum():.1%}",
)

st.sidebar.header("⚙️ Wine chemistry")
sample = pd.DataFrame(
    [
        {
            col: st.sidebar.slider(
                col.replace("_", " "),
                float(X[col].min()),
                float(X[col].max()),
                float(X[col].median()),
            )
            for col in X.columns
        }
    ]
)

point = project.transform(sample)[0]
proba = model.predict_proba(sample)[0]
predicted = model.classes_[int(np.argmax(proba))]

left, right = st.columns([1, 2])
with left:
    st.subheader("🎯 Prediction")
    st.success(f"**Cultivar {predicted}** · confidence {proba.max():.1%}")
    st.bar_chart(
        pd.Series(proba, index=[f"Cultivar {c}" for c in model.classes_], name="probability")
    )
    st.caption(f"Position on the map: {AXES[0]} = {point[0]:.2f}, {AXES[1]} = {point[1]:.2f}")

with right:
    st.subheader(f"🗺️ The data in {METHOD} space")
    Z = project.transform(X)
    pad = 1.0
    gx, gy = np.meshgrid(
        np.linspace(Z[:, 0].min() - pad, Z[:, 0].max() + pad, 250),
        np.linspace(Z[:, 1].min() - pad, Z[:, 1].max() + pad, 250),
    )
    regions = model.named_steps["clf"].predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)

    fig = go.Figure()
    fig.add_contour(
        x=gx[0], y=gy[:, 0], z=regions, showscale=False, opacity=0.18, hoverinfo="skip",
        contours=dict(coloring="fill"), line_width=0,
        colorscale=[[0, COLORS[0]], [0.5, COLORS[1]], [1, COLORS[2]]],
    )
    for color, cls in zip(COLORS, sorted(y.unique())):
        mask = (y == cls).to_numpy()
        fig.add_scatter(
            x=Z[mask, 0], y=Z[mask, 1], mode="markers", name=f"Cultivar {cls}",
            marker=dict(color=color, size=8, line=dict(width=0.5, color="white")),
        )
    fig.add_scatter(
        x=[point[0]], y=[point[1]], mode="markers", name="Your wine",
        marker=dict(symbol="star", size=20, color="black", line=dict(width=2, color="gold")),
    )
    fig.update_layout(
        template="plotly_white", height=480, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=f"{AXES[0]} ({explained[0]:.0%})", yaxis_title=f"{AXES[1]} ({explained[1]:.0%})",
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("📁 Dataset (first 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)
