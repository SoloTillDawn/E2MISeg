from e2miseg.evaluation.model_selection.summarize_results_in_one_json import summarize2
from e2miseg.paths import network_training_output_dir
from batchgenerators.utilities.file_and_folder_operations import *

if __name__ == "__main__":
    summary_output_folder = join(network_training_output_dir, "summary_jsons_fold0_new")
    maybe_mkdir_p(summary_output_folder)
    summarize2(['all'], output_dir=summary_output_folder, folds=(0,))

    results_csv = join(network_training_output_dir, "summary_fold0.csv")

    summary_files = subfiles(summary_output_folder, suffix='.json', join=False)

    with open(results_csv, 'w') as f:
        for s in summary_files:
            if s.find("ensemble") == -1:
                task, network, trainer, plans, validation_folder, folds = s.split("__")
            else:
                n1, n2 = s.split("--")
                n1 = n1[n1.find("ensemble_") + len("ensemble_") :]
                task = s.split("__")[0]
                network = "ensemble"
                trainer = n1
                plans = n2
                validation_folder = "none"
            folds = folds[:-len('.json')]
            results = load_json(join(summary_output_folder, s))
            results_mean = results['results']['mean']['mean']['Dice']
            results_median = results['results']['median']['mean']['Dice']
            f.write("%s,%s,%s,%s,%s,%02.4f,%02.4f\n" % (task,
                                            network, trainer, validation_folder, plans, results_mean, results_median))

    summary_output_folder = join(network_training_output_dir, "summary_jsons_new")
    maybe_mkdir_p(summary_output_folder)
    summarize2(['all'], output_dir=summary_output_folder)

    results_csv = join(network_training_output_dir, "summary_allFolds.csv")

    summary_files = subfiles(summary_output_folder, suffix='.json', join=False)

    with open(results_csv, 'w') as f:
        for s in summary_files:
            if s.find("ensemble") == -1:
                task, network, trainer, plans, validation_folder, folds = s.split("__")
            else:
                n1, n2 = s.split("--")
                n1 = n1[n1.find("ensemble_") + len("ensemble_") :]
                task = s.split("__")[0]
                network = "ensemble"
                trainer = n1
                plans = n2
                validation_folder = "none"
            folds = folds[:-len('.json')]
            results = load_json(join(summary_output_folder, s))
            results_mean = results['results']['mean']['mean']['Dice']
            results_median = results['results']['median']['mean']['Dice']
            f.write("%s,%s,%s,%s,%s,%02.4f,%02.4f\n" % (task,
                                            network, trainer, validation_folder, plans, results_mean, results_median))

