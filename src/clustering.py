import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "Glucose",
    "Insulin",
    "BMI",
    "Age",
    "DiabetesPedigreeFunction"
]

CLUSTER_FEATURES = FEATURES


def prepare_clustering_data(df):
    X = df[FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def evaluate_k_values(X_scaled, k_values=range(2, 11)):
    inertias = []
    silhouettes = []
    models = []

    for k in k_values:
        km = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = km.fit_predict(X_scaled)

        inertia = km.inertia_
        silhouette = silhouette_score(X_scaled, labels)

        inertias.append(inertia)
        silhouettes.append(silhouette)
        models.append(km)

    return inertias, silhouettes, models


def train_kmeans(X_scaled, n_clusters=2):
    km = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = km.fit_predict(X_scaled)

    return km, labels


def add_clusters(df, labels):
    df = df.copy()
    df["cluster"] = labels

    return df


def assign_risk_category(df):
    df = df.copy()

    means = df.groupby("cluster")["Glucose"].mean()
    high_risk_cluster = means.idxmax()

    df["risk_category"] = df["cluster"].map({
        high_risk_cluster: "high risk",
        1 - high_risk_cluster: "low risk"
    })

    return df