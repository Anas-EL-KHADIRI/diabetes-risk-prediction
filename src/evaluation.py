import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    report = classification_report(
        y_test,
        predictions
    )

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    return {
        "predictions": predictions,
        "confusion_matrix": cm,
        "classification_report": report,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_weighted": f1
    }


def plot_confusion_matrix(y_test, predictions, model_name):

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"confusion matrix - {model_name}")
    plt.xlabel("predicted")
    plt.ylabel("actual")

    plt.tight_layout()

    return plt


def compare_models(models, X_test, y_test):
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(
                y_test, y_pred,
                average="weighted",
                zero_division=0
            ), 4),
            "Recall": round(recall_score(
                y_test, y_pred,
                average="weighted",
                zero_division=0
            ), 4),
            "F1-score": round(f1_score(
                y_test, y_pred,
                average="weighted",
                zero_division=0
            ), 4),
            "Recall (High Risk)": round(recall_score(
                y_test, y_pred,
                pos_label="high risk",
                zero_division=0
            ), 4)
        })

    return pd.DataFrame(results).sort_values(
        by="F1-score",
        ascending=False
    )

