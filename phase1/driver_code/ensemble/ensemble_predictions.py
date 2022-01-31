import os
import pandas as pd
import numpy as np
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

    def predict(self, model_info: tuple) -> str:
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
    # smiles test compound
    # venetoclax
    # smiles = "CC1(C)CCC(CN2CCN(CC2)C2=CC=C(C(=O)NS(=O)(=O)C3=CC=C(NCC4CCOCC4)C(=C3)[N ]([O-])=O)C
    # (OC3=CN=C4NC=CC4=C3)=C2)=C(C1)C1=CC=C(Cl)C=C1"
    # JQ1
    smiles = "CC1=C(SC2=C1C(=NC(C3=NN=C(N32)C)CC(=O)O)C4=CC=C(C=C4)Cl)C"
    # acetaminophen
    # smiles = "CC(=O)NC1=CC=C(C=C1)O"
    # apabetalon
    # smiles = "COC1=CC2=C(C(=O)N=C(N2)C2=CC(C)=C(OCCO)C(C)=C2)C(OC)=C1"
    # idasanutline
    # smiles = "COC1=CC(=CC=C1NC(=O)[C@@H]1N[C@@H](CC(C)(C)C)[C@@](C#N)([C@H]1C1=CC=CC(Cl)=C1F)C1=C
    # C=C(Cl)C=C1F)C(O)=O"
    P = Prediction(
        data_root=local_root["data"],
        root=local_root["phase1"],
        input_file=input_filename,
        descriptors=descriptor_list,
        test_compound=smiles,
    )
    # model_filename = [(pk, model_filename)]
    model_filenames = [
        ("RF3", "RFF1L6P3EN2A.pkl"),
        ("RF4", "RFF1L6P3EN2B.pkl"),
        ("RF9", "RFF1L6P3GN2A.pkl"),
        ("LRG3", "LRGF1L6P3S2A.pkl"),
        ("LRG4", "LRGF1L6P3S2B.pkl"),
        ("LRG9", "LRGF1L6P3S5A.pkl"),
        ("SVM6", "SVMF1L6P3K3B.pkl"),
        ("SVM7", "SVMF1L6P3K4A.pkl"),
        ("ensemble1", "ensemble1.pkl"),
    ]
    for i in model_filenames:
        prediction = P.predict(i)
        print(f"prediction {i[0]}: {prediction}")
