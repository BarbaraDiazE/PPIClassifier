import os
import pandas as pd
import numpy as np
import json
import joblib

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def get_ecfp4(smiles):
    ms = [Chem.MolFromSmiles(element) for element in smiles]
    fp = [AllChem.GetMorganFingerprintAsBitVect(x, 2) for x in ms]
    return fp


class Prediction:
    def __init__(
        self,
        data_root: str,
        root: str,
        input_file: str,
        descriptors: list,
        test_compound: str,
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
        self.root = root
        self.data = data
        self.descriptors = descriptors
        self.numerical_data = data.drop(ids, axis=1)
        self.test_compound = test_compound

    def explicit_descriptor(self):
        fp = get_ecfp4([self.test_compound])
        output = []
        for f in fp:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(f, arr)
            output.append(arr)
        return np.asarray(output)

    def predict(self, model_info: tuple, test_smiles) -> str:
        test_smiles = self.explicit_descriptor()
        model = joblib.load(
            os.path.join(
                self.root,
                "results",
                "trained_results",
                "trained_models",
                (model_info[1]),
            )
        )
        x = model.predict(test_smiles)
        prediction = str()
        if x[0] == 0:
            prediction = "Inactive"
        elif x[0] == 1:
            prediction = "Active"
        return prediction


if __name__ == "__main__":
    from phase1.support_functions.support_descriptors import get_numerical_descriptors
    from phase1.support_functions.vars import local_root

    input_filename = "dataset_ecfp4.csv"
    descriptor_list = get_numerical_descriptors(input_filename)
    json_path = os.getcwd()
    f = open(os.path.join(json_path, "test_compounds.json"))
    test_compounds = json.load(f)
    # get smiles value
    test_smiles_ = test_compounds["venetoclax"]
    P = Prediction(
        data_root=local_root["data"],
        root=local_root["phase1"],
        input_file=input_filename,
        descriptors=descriptor_list,
        test_compound=test_smiles_,
    )
    # model_filename = [(pk, model_filename)]
    model_filenames = [
        ("RF4", "RFF1L6P3EN2B.pkl"),
        ("RF9", "RFF1L6P3GN2A.pkl"),
        ("LRG1", "LRGF1L6P3S1A.pkl"),
        ("LRG4", "LRGF1L6P3S2B.pkl"),
        ("LRG9", "LRGF1L6P3S5A.pkl"),
        ("SVM6", "SVMF1L6P3K3B.pkl"),
        ("SVM7", "SVMF1L6P3K4A.pkl"),
        ("ensemble1", "ensemble1.pkl"),
    ]

    for i in model_filenames:
        prediction = P.predict(i, test_smiles_)
        print(f"prediction {i[0]}: {prediction}")
