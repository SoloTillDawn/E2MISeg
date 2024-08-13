import os
import shutil
from batchgenerators.utilities.file_and_folder_operations import subdirs, subfiles


def crawl_and_copy(current_folder, out_folder, prefix="xwp_", suffix="summary.json"):
    s = subdirs(current_folder, join=False)
    f = subfiles(current_folder, join=False)
    f = [i for i in f if i.endswith(suffix)]
    if current_folder.find("fold0") != -1:
        for fl in f:
            shutil.copy(os.path.join(current_folder, fl), os.path.join(out_folder, prefix+fl))
    for su in s:
        if prefix == "":
            add = su
        else:
            add = "__" + su
        crawl_and_copy(os.path.join(current_folder, su), out_folder, prefix=prefix+add)


if __name__ == "__main__":
    from e2miseg.paths import network_training_output_dir
    output_folder = "......../pmunet/output_acdc"
    crawl_and_copy(network_training_output_dir, output_folder)
    from e2miseg.evaluation.add_mean_dice_to_json import run_in_folder
    run_in_folder(output_folder)
