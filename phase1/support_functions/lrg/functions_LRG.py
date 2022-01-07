import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    f1_score,
    roc_auc_score,
    auc,
    roc_curve,
    confusion_matrix,
    recall_score,
)


def plot_roc(output_ref, y_test, y_score, local_root) -> float:
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    roc_root = f"{local_root}/results/trained_results/roc_plot"
    output_filename = output_ref + "_roc_data.csv"
    roc_df = pd.DataFrame.from_dict({"fpr": fpr, "tpr": tpr})
    roc_df.to_csv(os.path.join(roc_root, "roc_data", output_filename), sep=",")
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color="red", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    output_filename = f"{output_ref}.png"
    plt.savefig(os.path.join(roc_root, output_filename))
    # plt.show()
    return roc_auc


def lrg_report(
    output_reference: str,
    data: pd.DataFrame,
    parameters: dict,
    y_test: np.array,
    predictions: np.array,
    descriptors,
    attributes: dict,
    roc_auc,
    local_root,
):
    df = pd.DataFrame({"Descriptors": descriptors, "Coeff": np.around(attributes["Coeff"], 2)})
    df = df.sort_values(by="Coeff", ascending=False)
    coeff_filename_df = f"{output_reference}_coeff.csv"
    coeff_output_root = f"{local_root}/results/trained_results/models_reports/coeff_reports"
    df.to_csv(os.path.join(coeff_output_root, coeff_filename_df), sep=",")
    report_data = {
        "Method": parameters["Method"],
        "Class weight": parameters["class weight"],
        "Solver": parameters["solver"],
        "Libraries": " ".join(map(str, list(data.library.unique()))),
        "Test fraction": parameters["fraction"],
        "Descriptors": " ".join(map(str, descriptors)),
        "Coeff": " ".join(map(str, list(np.around(np.array(df["Coeff"]), 2)))),
        "Classes": " ".join(map(str, list(attributes["classes"]))),
        "Interception": round(attributes["inter"][0], 2),
        "N iter": attributes["iter"][0],
        "Accuracy": round(accuracy_score(y_test, predictions), 2),
        "Balanced Accuracy": round(balanced_accuracy_score(y_test, predictions), 2),
        "Precision": round(precision_score(y_test, predictions), 2),
        "F1": round(f1_score(y_test, predictions), 2),
        "ROC AUC score": round(roc_auc_score(y_test, predictions), 2),
        "AUC": round(roc_auc, 2),
        "Confusion matrix": confusion_matrix(y_test, predictions),
        "Recall": round(recall_score(y_test, predictions), 2),
    }
    report = pd.DataFrame.from_dict(report_data, orient="index", columns=["value"])
    output_report_name = f"{output_reference}.csv"
    output_root = f"{local_root}/results/trained_results/models_reports"
    report.to_csv(os.path.join(output_root, output_report_name), sep=",")
