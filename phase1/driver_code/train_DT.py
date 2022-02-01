"""Instructions to train  DT from different representation"""

from phase1.support_functions.dt.DT import DT
from phase1.support_functions.generate_output_model_name import get_dt_output
from phase1.support_functions.dt.DT import custom_descriptors


def execute(
    root_data: str,
    local_root: str,
    input_file: str,
    target: str,
    fp: str,
    fraction: float,
    criterion: str,
    max_depth,
    class_weight: str,
):
    descriptor_list = custom_descriptors
    a = DT(root_data, local_root, input_file, target, descriptor_list, fraction)
    output_reference = get_dt_output(fp, "L6", fraction, criterion, class_weight)
    a.report(criterion, max_depth, class_weight, output_reference)


"""set params to train"""
if __name__ == "__main__":
    from phase1.support_functions.vars import local_root, proportion_list

    # fp_list = ["ECFP4", "ECFP6", "MACCSKEYS", "AtomPairs"]
    descriptor_list = ["descriptors"]
    criterion_list = ["gini", "entropy"]
    balanced_list = ["balanced", None]
    max_depth_list = [None]
    for i in range(len(descriptor_list)):
        filename = f'{"dataset_"}{descriptor_list[i].lower()}{".csv"}'
        for j in range(len(proportion_list)):
            for c in range(len(criterion_list)):
                for n in range(len(balanced_list)):
                    execute(
                        root_data=local_root["data"],
                        local_root=local_root["phase1"],
                        input_file=filename,
                        target="PPI",
                        fp=descriptor_list[i],
                        fraction=proportion_list[j],
                        criterion=criterion_list[c],
                        # max_depth=None,
                        max_depth=5,
                        class_weight=balanced_list[n],
                    )
