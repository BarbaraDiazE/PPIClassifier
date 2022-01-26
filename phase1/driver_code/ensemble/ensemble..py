import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from phase1.support_functions.ensemble.functions_ensemble import ensemble_report
from phase1.support_functions.ensemble.functions_ensemble import save_ensemble


class Ensemble:
    def __init__(
        self,
        data_root: str,
        local_root: str,
        input_file: str,
        target: str,
        descriptors: list,
        fraction: float,
    ):
        data = pd.read_csv(os.path.join(data_root, input_file), index_col="Unnamed: 0")
        self.root_data = data_root
        ids = [
            "ipp_id",
            "chembl_id",
            "SMILES",
            "library",
            "PPI family",
            "PPI",
        ]
        print(
            "\n",
            "PPI types: ",
            data.PPI.unique(),
            "\n",
            "Libraries types: ",
            data.library.unique(),
            "\n",
            "Total compounds number: ",
            data.shape[0],
        )
        self.local_root = local_root
        self.target = target
        self.descriptors = descriptors
        self.fraction = fraction
        self.numerical_data = data.drop(ids, axis=1)
        self.data = data
        print("Ensemble class instanced")

    def train_model(self, solver):
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression

        models = [
            (
                "lr",
                LogisticRegression(
                    solver=solver,
                    fit_intercept=True,
                    # class_weight=class_weight,
                    random_state=1992,
                    n_jobs=2,
                ),
            ),
            # ('rf', RandomForestClassifier()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    probability=True,
                    # class_weight="balanced",
                    random_state=1992,
                ),
            ),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    probability=True,
                    # class_weight="balanced",
                    random_state=1992,
                ),
            ),
        ]
        ensemble = VotingClassifier(estimators=models, voting="soft")
        y = np.array(self.data[self.target])
        y = label_binarize(y, classes=["No", "Yes"])
        y = np.reshape(y, int(y.shape[0]))
        numerical_data = pd.DataFrame(StandardScaler().fit_transform(self.data[self.descriptors]))
        x_train, x_test, y_train, y_test = train_test_split(
            numerical_data, y, test_size=self.fraction, random_state=1992
        )
        return ensemble.fit(x_train, y_train), x_test, y_test

    @classmethod
    def get_predictions(cls, model, x_test, y_test):
        prediction_data = {
            "predictions": model.predict(x_test),
            # "y_score": model.decision_function(x_test),
            "x_text": x_test,
            "y_test": y_test,
        }
        return prediction_data

    # @classmethod
    # def get_params(cls, kernel, class_weight, fraction):
    #     parameters = {
    #         "Method": "Linear Regression",
    #         "Class weight": class_weight,
    #         "kernel": kernel,
    #         "fraction": fraction * 100,
    #     }
    #     return parameters
    #
    def report(self, solver: str, output_reference: str):
        ensemble, x_test, y_test = self.train_model(solver)
        print("report method")
        prediction_data = self.get_predictions(ensemble, x_test, y_test)
        #     roc_auc = plot_roc(
        #         output_reference,
        #         prediction_data["y_test"],
        #         prediction_data["y_score"],
        #         self.local_root,
        #     )
        ensemble_report(
            output_reference=output_reference,
            data=self.data,
            #         parameters=self.get_params(kernel, class_weight, self.fraction),
            y_test=prediction_data["y_test"],
            predictions=prediction_data["predictions"],
            descriptors=self.descriptors,
            #         attributes=self.get_attributes(model, kernel),
            #         roc_auc=roc_auc,
            local_root=self.local_root,
        )
        save_ensemble(ensemble, output_reference, self.local_root)


if __name__ == "__main__":
    from phase1.support_functions.support_descriptors import get_numerical_descriptors
    from phase1.support_functions.vars import local_root

    input_filename = "dataset_ecfp4.csv"
    descriptor_list = get_numerical_descriptors(input_filename)
    E = Ensemble(
        data_root=local_root["data"],
        local_root=local_root["phase1"],
        input_file=input_filename,
        target="PPI",
        descriptors=descriptor_list,
        fraction=0.2,
    )
    # model, x_test, y_test = E.train_model(solver="newton-cg")
    E.report(solver="newton-cg", output_reference="ensemble")
    # output_reference = get_lrg_output(fp, "L6", fraction, solver, balanced)
    # a.report(solver, balanced, output_reference)
