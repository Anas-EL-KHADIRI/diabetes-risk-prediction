import os
import platform
import sklearn
import pandas as pd
import numpy as np
import imblearn

from matplotlib import pyplot as plt
import mlflow
import mlflow.sklearn
import joblib
import tempfile

from mlflow.models import infer_signature

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def setup_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("diabetes-risk")


def log_confusion_matrix(y_test, predictions, model_name):
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=["high risk", "low risk"],
        cmap="Blues"
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    mlflow.log_figure(
        plt.gcf(),
        "confusion_matrix.png"
    )

    plt.close()


def log_classification_run(
    model,
    model_name,
    X_train,
    X_test,
    y_test,
    predictions,
    params=None
):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_dict(
            {"features": list(X_train.columns)},
            "features.json"
        )

        if params:
            mlflow.log_params(params)

        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, predictions),
            "precision_weighted": precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),
            "recall_weighted": recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),
            "f1_weighted": f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )
        })

        mlflow.log_metric(
            "recall_high_risk",
            recall_score(
                y_test,
                predictions,
                average="binary",
                pos_label="high risk",
                zero_division=0
            )
        )
        signature = infer_signature(
            X_train,
            model.predict(X_train)
        )

        input_example = X_train.head(5)
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "imblearn.over_sampling._random_over_sampler.RandomOverSampler",
                "imblearn.pipeline.Pipeline",
                "sklearn.tree._tree.Tree",
                "sklearn.metrics._dist_metrics.EuclideanDistance64",
                "sklearn.neighbors._kd_tree.KDTree"
            ],
            signature=signature,
            input_example=input_example,
        )
        log_confusion_matrix(
            y_test,
            predictions,
            model_name
        )
        mlflow.log_params({
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "imblearn_version": imblearn.__version__,
            "mlflow_version": mlflow.__version__
        })
    return model_info.model_uri


def log_clustering_run(
    kmeans,
    scaler,
    k,
    inertia,
    silhouette,
    features=None
):
    with mlflow.start_run(run_name=f"KMeans-{k}"):
        mlflow.log_param("model", "KMeans")
        mlflow.log_param("n_clusters", k)
        mlflow.log_param("scaler", "StandardScaler")

        if features is not None:
            mlflow.log_dict(
                {"features": list(features)},
                "features.json"
            )

        mlflow.log_metrics({
            "inertia": float(inertia),
            "silhouette_score": float(silhouette)
        })

        mlflow.sklearn.log_model(
            kmeans,
            name="kmeans_model"
        )

        mlflow.sklearn.log_model(
            scaler,
            name="scaler"
        )

        kmeans_path = os.path.join(tempfile.gettempdir(), "kmeans.joblib")
        scaler_path = os.path.join(tempfile.gettempdir(), "scaler.joblib")
        joblib.dump(kmeans, kmeans_path)
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(kmeans_path, "artifacts")
        mlflow.log_artifact(scaler_path, "artifacts")