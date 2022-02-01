import os
import pandas as pd
import numpy as np

from phase1.support_functions.json_functions import read_id_json
from phase1.support_functions.metrics import metrics


class GetReportValues:
    def __init__(self, local_root: str):
        self.local_root = local_root

    def get_reports_filenames(self, algorithm_name: str):
        json_filename = f"{algorithm_name.upper()}_models_id.json"
        json_root = f"{self.local_root}/results/trained_results/id_models"
        data_dict = read_id_json(json_root, json_filename)
        numerical_id = [k for k in data_dict]
        pk_models = [data_dict[k] for k in data_dict]
        files_list = [pk + ".csv" for pk in pk_models]
        return numerical_id, pk_models, files_list

    def get_report_value(self, metric: str, filename: str) -> float:
        data_root = f"{self.local_root}/results/trained_results/models_reports"
        df = pd.read_csv(os.path.join(data_root, filename), sep=",", index_col="Unnamed: 0")
        value = float(df.loc[metric].value)
        return value

    @classmethod
    def get_numerical_id(cls, filename, algorithm_name) -> str:
        """get equivalence filename and numerical_id"""
        json_root = f"{local_root}/results/trained_results/id_models"
        json_filename = f"{algorithm_name.upper()}_models_id.json"
        data_dict = read_id_json(json_root, json_filename)
        for numerical_id, filename_ in data_dict.items():
            if filename_ == filename.replace(".csv", ""):
                return numerical_id
            else:
                continue


def save_df(best_models_names, id_metrics, algorithm_name, quartile, local_root):
    best_models_ = np.array([best_models_names])
    df = pd.DataFrame(
        data=best_models_.transpose(),
        columns=["FK model"],
    )
    output_filename = f"{id_metrics}_{algorithm_name}_{quartile}_best_models.csv"
    output_root = f"{local_root}/results/metrics_results/filtered_results"
    df.to_csv(os.path.join(output_root, output_filename))


class ModelFiltering(GetReportValues):
    def __init__(self, algorithm_name: str, quartile):
        self.algorithm = algorithm_name
        self.quartile = quartile
        super().__init__(local_root)

    def get_q3(self, metric: str):
        df_filename = f"{metric.lower()}_{self.algorithm}_summary.csv"
        metrics_root = f"{local_root}/results/metrics_results/metrics_reports"
        df = pd.read_csv(os.path.join(metrics_root, df_filename), index_col="Unnamed: 0")
        df[metric] = [float(i) for i in list(df["Value"])]
        stat_df = df.describe()
        q3 = stat_df.loc["75%", metric]
        return q3

    def get_q2(self, metric: str):
        df_filename = f"{metric.lower()}_{self.algorithm}_summary.csv"
        metrics_root = f"{local_root}/results/metrics_results/metrics_reports"
        df = pd.read_csv(os.path.join(metrics_root, df_filename), index_col="Unnamed: 0")
        df[metric] = [float(i) for i in list(df["Value"])]
        stat_df = df.describe()
        q2 = stat_df.loc["50%", metric]
        return q2

    def get_best_metrics_models(self, metric: str):
        """Those with metrics higher than q3 for a single metric"""
        numerical_id, pk_models, files_list = self.get_reports_filenames(self.algorithm)
        q = str()
        if self.quartile == "Q2":
            q = self.get_q2(metric)
        elif self.quartile == "Q3":
            q = self.get_q3(metric)
        best_models = dict()
        for _ in files_list:
            _val = self.get_report_value(metric, _)
            if _val >= q:
                numerical_id_ = self.get_numerical_id(_, self.algorithm)
                best_models[numerical_id_] = _val
            else:
                continue
        return best_models, numerical_id

    def get_best_models(self, metrics_: list, id_: str):
        """those present in all metrics filtered list"""
        num_id = precision_bm = accuracy_bm = recall_bm = list()
        for metric in metrics_:
            if metric.lower() == "precision":
                precision_bm, num_id = self.get_best_metrics_models(metric)
            if metric.lower() == "accuracy":
                accuracy_bm, num_id = self.get_best_metrics_models(metric)
            if metric.lower() == "recall":
                recall_bm, num_id = self.get_best_metrics_models(metric)
        # comment or uncomment metrics to perform analysis
        best_models_names = [i for i in num_id if i in accuracy_bm and i in precision_bm]
        # best_models_names = [i for i in num_id if i in accuracy_bm and i in recall_bm]
        save_df(best_models_names, id_, self.algorithm, self.quartile, self.local_root)


if __name__ == "__main__":
    from phase1.support_functions.vars import local_root

    local_root = local_root["phase1"]
    """
    id_ is manually setted according metrics
    "M1" for accuracy and precision
    "M2" for accuracy and recall
    """
    id_ = "M2"
    algorithms = ["svm", "lrg", "rf", "dt"]
    for algorithm in algorithms:
        ModelFiltering(algorithm, "Q2").get_best_models(metrics, id_)
        ModelFiltering(algorithm, "Q3").get_best_models(metrics, id_)
