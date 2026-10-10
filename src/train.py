
import os
import joblib
import mlflow
from mlflow import MlflowClient

from preprocessing import load_data, preprocess_data

from clustering import (
    FEATURES,
    prepare_clustering_data,
    evaluate_k_values,
    train_kmeans,
    add_clusters,
    assign_risk_category
)

from classification import (
    build_models,
    split_data,
    train_models,
    tune_logistic_regression
)

from evaluation import evaluate_model, compare_models

from mlflow_tracking import (
    setup_mlflow,
    log_classification_run,
    log_clustering_run
)


def main():
    setup_mlflow()
    df = load_data()
    df = preprocess_data(df)
    print(df.head())
    X_scaled, scaler = prepare_clustering_data(df)
    inertias, silhouettes, clustering_models = evaluate_k_values(
        X_scaled
    )

    print("Inertias:", inertias)
    print("Silhouette scores:", silhouettes)

    for k, inertia, silhouette, km_model in zip(
        range(2, 11),
        inertias,
        silhouettes,
        clustering_models
    ):
        log_clustering_run(
            kmeans=km_model,
            scaler=scaler,
            k=k,
            inertia=inertia,
            silhouette=silhouette,
            features=FEATURES
        )

    km, labels = train_kmeans(X_scaled, n_clusters=2)
    df = add_clusters(df, labels)

    cluster_analysis = df.groupby("cluster")[
        FEATURES
    ].mean()

    print("\nCluster analysis:")
    print(cluster_analysis)

    df = assign_risk_category(df)
    X_train, X_test, y_train, y_test = split_data(df)

    models = build_models()

    trained_models = train_models(
        models,
        X_train,
        y_train
    )

    candidates = {}

    for name, model in trained_models.items():

        results = evaluate_model(
            model,
            X_test,
            y_test
        )

        model_uri = log_classification_run(
            model=model,
            model_name=name,
            X_train=X_train,
            X_test=X_test,
            y_test=y_test,
            predictions=results["predictions"]
        )

        candidates[name] = {
            "model": model,
            "model_uri": model_uri,
            "f1_weighted": results["f1_weighted"]
        }

        print(f"\n{name}")
        print("Accuracy:", results["accuracy"])
        print("Precision:", results["precision"])
        print("Recall:", results["recall"])
        print("F1:", results["f1_weighted"])
        print("Confusion Matrix:")
        print(results["confusion_matrix"])
        print(results["classification_report"])

    lr_search = tune_logistic_regression(
        X_train,
        y_train
    )

    print("\nBest Logistic Regression parameters:")
    print(lr_search.best_params_)

    best_lr = lr_search.best_estimator_

    results = evaluate_model(
        best_lr,
        X_test,
        y_test
    )

    model_uri = log_classification_run(
        model=best_lr,
        model_name="Tuned Logistic Regression",
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        predictions=results["predictions"],
        params=lr_search.best_params_
    )

    candidates["Tuned Logistic Regression"] = {
        "model": best_lr,
        "model_uri": model_uri,
        "f1_weighted": results["f1_weighted"]
    }

    print("\nTuned Logistic Regression")
    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1:", results["f1_weighted"])
    print("Confusion Matrix:")
    print(results["confusion_matrix"])
    print(results["classification_report"])

    trained_models["Tuned Logistic Regression"] = best_lr

    comparison = compare_models(
        trained_models,
        X_test,
        y_test
    )

    print("\nModel comparison:")
    print(comparison.to_string(index=False))

    best_name = max(
        candidates,
        key=lambda name: candidates[name]["f1_weighted"]
    )

    best_candidate = candidates[best_name]

    print("\nBest model:", best_name)
    print("Best weighted F1:", best_candidate["f1_weighted"])

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_candidate["model"], "models/pipeline.joblib")
    print("Saved champion pipeline to models/pipeline.joblib")

    registered_model_name = "diabetes-risk-classifier"

    registered_model = mlflow.register_model(
        model_uri=best_candidate["model_uri"],
        name=registered_model_name
    )

    client = MlflowClient()

    client.set_registered_model_alias(
        name=registered_model_name,
        alias="champion",
        version=registered_model.version
    )

    client.transition_model_version_stage(
        name=registered_model_name,
        version=registered_model.version,
        stage="Production",
        archive_existing_versions=True
    )

    print("\nModel registered successfully.")
    print("Model:", registered_model_name)
    print("Version:", registered_model.version)
    print("Alias: champion")
    print("Stage: Production")
    print("Selected algorithm:", best_name)


if __name__ == "__main__":
    main()