import ast
from copy import deepcopy
from multiprocessing.pool import Pool

import numpy as np
from e2miseg.run.default_configuration import default_num_threads
from e2miseg.evaluation.evaluator import aggregate_scores
from scipy.ndimage import label
import SimpleITK as sitk
from e2miseg.utilities.sitk_stuff import copy_geometry
from batchgenerators.utilities.file_and_folder_operations import *
import shutil


def load_remove_save(input_file: str, output_file: str, for_which_classes: list,
                     minimum_valid_object_size: dict = None):
    img_in = sitk.ReadImage(input_file)
    img_npy = sitk.GetArrayFromImage(img_in)
    volume_per_voxel = float(np.prod(img_in.GetSpacing(), dtype=np.float64))

    image, largest_removed, kept_size = remove_all_but_the_largest_connected_component(img_npy, for_which_classes,
                                                                                       volume_per_voxel,
                                                                                       minimum_valid_object_size)
    # print(input_file, "kept:", kept_size)
    img_out_itk = sitk.GetImageFromArray(image)
    img_out_itk = copy_geometry(img_out_itk, img_in)
    sitk.WriteImage(img_out_itk, output_file)
    return largest_removed, kept_size


def remove_all_but_the_largest_connected_component(image: np.ndarray, for_which_classes: list, volume_per_voxel: float,
                                                   minimum_valid_object_size: dict = None):

    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes, "cannot remove background"
    largest_removed = {}
    kept_size = {}
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            maximum_size = max(object_sizes.values())
            kept_size[c] = maximum_size

            for object_id in range(1, num_objects + 1):
                # we only remove objects that are not the largest
                if object_sizes[object_id] != maximum_size:
                    # we only remove objects that are smaller than minimum_valid_object_size
                    remove = True
                    if minimum_valid_object_size is not None:
                        remove = object_sizes[object_id] < minimum_valid_object_size[c]
                    if remove:
                        image[(lmap == object_id) & mask] = 0
                        if largest_removed[c] is None:
                            largest_removed[c] = object_sizes[object_id]
                        else:
                            largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
    return image, largest_removed, kept_size


def load_postprocessing(json_file):

    a = load_json(json_file)
    if 'min_valid_object_sizes' in a.keys():
        min_valid_object_sizes = ast.literal_eval(a['min_valid_object_sizes'])
    else:
        min_valid_object_sizes = None
    return a['for_which_classes'], min_valid_object_sizes


def determine_postprocessing(base, gt_labels_folder, raw_subfolder_name="validation_raw",
                             temp_folder="temp",
                             final_subf_name="validation_final", processes=default_num_threads,
                             dice_threshold=0, debug=False,
                             advanced_postprocessing=False,
                             pp_filename="postprocessing.json"):

    classes = [int(i) for i in load_json(join(base, raw_subfolder_name, "summary.json"))['results']['mean'].keys() if
               int(i) != 0]

    folder_all_classes_as_fg = join(base, temp_folder + "_allClasses")
    folder_per_class = join(base, temp_folder + "_perClass")

    if isdir(folder_all_classes_as_fg):
        shutil.rmtree(folder_all_classes_as_fg)
    if isdir(folder_per_class):
        shutil.rmtree(folder_per_class)

    # multiprocessing rules
    p = Pool(processes)

    assert isfile(join(base, raw_subfolder_name, "summary.json")), "join(base, raw_subfolder_name) does not " \
                                                                   "contain a summary.json"

    # these are all the files we will be dealing with
    fnames = subfiles(join(base, raw_subfolder_name), suffix=".nii.gz", join=False)

    # make output and temp dir
    maybe_mkdir_p(folder_all_classes_as_fg)
    maybe_mkdir_p(folder_per_class)
    maybe_mkdir_p(join(base, final_subf_name))

    pp_results = {}
    pp_results['dc_per_class_raw'] = {}
    pp_results['dc_per_class_pp_all'] = {}  # dice scores after treating all foreground classes as one
    pp_results['dc_per_class_pp_per_class'] = {}  # dice scores after removing everything except larges cc
    # independently for each class after we already did dc_per_class_pp_all
    pp_results['for_which_classes'] = []
    pp_results['min_valid_object_sizes'] = {}


    validation_result_raw = load_json(join(base, raw_subfolder_name, "summary.json"))['results']
    pp_results['num_samples'] = len(validation_result_raw['all'])
    validation_result_raw = validation_result_raw['mean']

    if advanced_postprocessing:
        results = []
        for f in fnames:
            predicted_segmentation = join(base, raw_subfolder_name, f)
            # now remove all but the largest connected component for each class
            output_file = join(folder_all_classes_as_fg, f)
            results.append(p.starmap_async(load_remove_save, ((predicted_segmentation, output_file, (classes,)),)))

        results = [i.get() for i in results]

        # aggregate max_size_removed and min_size_kept
        max_size_removed = {}
        min_size_kept = {}
        for tmp in results:
            mx_rem, min_kept = tmp[0]
            for k in mx_rem:
                if mx_rem[k] is not None:
                    if max_size_removed.get(k) is None:
                        max_size_removed[k] = mx_rem[k]
                    else:
                        max_size_removed[k] = max(max_size_removed[k], mx_rem[k])
            for k in min_kept:
                if min_kept[k] is not None:
                    if min_size_kept.get(k) is None:
                        min_size_kept[k] = min_kept[k]
                    else:
                        min_size_kept[k] = min(min_size_kept[k], min_kept[k])

        print("foreground vs background, smallest valid object size was", min_size_kept[tuple(classes)])
        print("removing only objects smaller than that...")

    else:
        min_size_kept = None

    # we need to rerun the step from above, now with the size constraint
    pred_gt_tuples = []
    results = []
    # first treat all foreground classes as one and remove all but the largest foreground connected component
    for f in fnames:
        predicted_segmentation = join(base, raw_subfolder_name, f)
        # now remove all but the largest connected component for each class
        output_file = join(folder_all_classes_as_fg, f)
        results.append(
            p.starmap_async(load_remove_save, ((predicted_segmentation, output_file, (classes,), min_size_kept),)))
        pred_gt_tuples.append([output_file, join(gt_labels_folder, f)])

    _ = [i.get() for i in results]

    # evaluate postprocessed predictions
    _ = aggregate_scores(pred_gt_tuples, labels=classes,
                         json_output_file=join(folder_all_classes_as_fg, "summary.json"),
                         json_author="Fabian", num_threads=processes)

    validation_result_PP_test = load_json(join(folder_all_classes_as_fg, "summary.json"))['results']['mean']

    for c in classes:
        dc_raw = validation_result_raw[str(c)]['Dice']
        dc_pp = validation_result_PP_test[str(c)]['Dice']
        pp_results['dc_per_class_raw'][str(c)] = dc_raw
        pp_results['dc_per_class_pp_all'][str(c)] = dc_pp

    # true if new is better
    do_fg_cc = False
    comp = [pp_results['dc_per_class_pp_all'][str(cl)] > (pp_results['dc_per_class_raw'][str(cl)] + dice_threshold) for
            cl in classes]
    before = np.mean([pp_results['dc_per_class_raw'][str(cl)] for cl in classes])
    after = np.mean([pp_results['dc_per_class_pp_all'][str(cl)] for cl in classes])
    print("Foreground vs background")
    print("before:", before)
    print("after: ", after)
    if any(comp):
        # at least one class improved - yay!
        # now check if another got worse
        # true if new is worse
        any_worse = any(
            [pp_results['dc_per_class_pp_all'][str(cl)] < pp_results['dc_per_class_raw'][str(cl)] for cl in classes])
        if not any_worse:
            pp_results['for_which_classes'].append(classes)
            if min_size_kept is not None:
                pp_results['min_valid_object_sizes'].update(deepcopy(min_size_kept))
            do_fg_cc = True
            print("Removing all but the largest foreground region improved results!")
            print('for_which_classes', classes)
            print('min_valid_object_sizes', min_size_kept)
    else:
        # did not improve things - don't do it
        pass

    if len(classes) > 1:
        # now depending on whether we do remove all but the largest foreground connected component we define the source dir
        # for the next one to be the raw or the temp dir
        if do_fg_cc:
            source = folder_all_classes_as_fg
        else:
            source = join(base, raw_subfolder_name)

        if advanced_postprocessing:
            # now run this for each class separately
            results = []
            for f in fnames:
                predicted_segmentation = join(source, f)
                output_file = join(folder_per_class, f)
                results.append(p.starmap_async(load_remove_save, ((predicted_segmentation, output_file, classes),)))

            results = [i.get() for i in results]

            # aggregate max_size_removed and min_size_kept
            max_size_removed = {}
            min_size_kept = {}
            for tmp in results:
                mx_rem, min_kept = tmp[0]
                for k in mx_rem:
                    if mx_rem[k] is not None:
                        if max_size_removed.get(k) is None:
                            max_size_removed[k] = mx_rem[k]
                        else:
                            max_size_removed[k] = max(max_size_removed[k], mx_rem[k])
                for k in min_kept:
                    if min_kept[k] is not None:
                        if min_size_kept.get(k) is None:
                            min_size_kept[k] = min_kept[k]
                        else:
                            min_size_kept[k] = min(min_size_kept[k], min_kept[k])

            print("classes treated separately, smallest valid object sizes are")
            print(min_size_kept)
            print("removing only objects smaller than that")
        else:
            min_size_kept = None

        # rerun with the size thresholds from above
        pred_gt_tuples = []
        results = []
        for f in fnames:
            predicted_segmentation = join(source, f)
            output_file = join(folder_per_class, f)
            results.append(p.starmap_async(load_remove_save, ((predicted_segmentation, output_file, classes, min_size_kept),)))
            pred_gt_tuples.append([output_file, join(gt_labels_folder, f)])

        _ = [i.get() for i in results]

        # evaluate postprocessed predictions
        _ = aggregate_scores(pred_gt_tuples, labels=classes,
                             json_output_file=join(folder_per_class, "summary.json"),
                             json_author="Fabian", num_threads=processes)

        if do_fg_cc:
            old_res = deepcopy(validation_result_PP_test)
        else:
            old_res = validation_result_raw

        # these are the new dice scores
        validation_result_PP_test = load_json(join(folder_per_class, "summary.json"))['results']['mean']

        for c in classes:
            dc_raw = old_res[str(c)]['Dice']
            dc_pp = validation_result_PP_test[str(c)]['Dice']
            pp_results['dc_per_class_pp_per_class'][str(c)] = dc_pp
            print(c)
            print("before:", dc_raw)
            print("after: ", dc_pp)

            if dc_pp > (dc_raw + dice_threshold):
                pp_results['for_which_classes'].append(int(c))
                if min_size_kept is not None:
                    pp_results['min_valid_object_sizes'].update({c: min_size_kept[c]})
                print("Removing all but the largest region for class %d improved results!" % c)
                print('min_valid_object_sizes', min_size_kept)
    else:
        print("Only one class present, no need to do each class separately as this is covered in fg vs bg")

    if not advanced_postprocessing:
        pp_results['min_valid_object_sizes'] = None

    print("/postprocessing/connected_components.py:后处理完成")
    print("for which classes:")
    print(pp_results['for_which_classes'])
    print("min_object_sizes")
    print(pp_results['min_valid_object_sizes'])

    pp_results['validation_raw'] = raw_subfolder_name
    pp_results['validation_final'] = final_subf_name

    # now that we have a proper for_which_classes, apply that
    pred_gt_tuples = []
    results = []
    for f in fnames:
        predicted_segmentation = join(base, raw_subfolder_name, f)

        # now remove all but the largest connected component for each class
        output_file = join(base, final_subf_name, f)
        results.append(p.starmap_async(load_remove_save, (
            (predicted_segmentation, output_file, pp_results['for_which_classes'],
             pp_results['min_valid_object_sizes']),)))

        pred_gt_tuples.append([output_file,
                               join(gt_labels_folder, f)])

    _ = [i.get() for i in results]
    # evaluate postprocessed predictions
    _ = aggregate_scores(pred_gt_tuples, labels=classes,
                         json_output_file=join(base, final_subf_name, "summary.json"),
                         json_author="Fabian", num_threads=processes)

    pp_results['min_valid_object_sizes'] = str(pp_results['min_valid_object_sizes'])

    save_json(pp_results, join(base, pp_filename))

    # delete temp
    if not debug:
        shutil.rmtree(folder_per_class)
        shutil.rmtree(folder_all_classes_as_fg)

    p.close()
    p.join()
    print("done")


def apply_postprocessing_to_folder(input_folder: str, output_folder: str, for_which_classes: list,
                                   min_valid_object_size:dict=None, num_processes=8):

    maybe_mkdir_p(output_folder)
    p = Pool(num_processes)
    nii_files = subfiles(input_folder, suffix=".nii.gz", join=False)
    input_files = [join(input_folder, i) for i in nii_files]
    out_files = [join(output_folder, i) for i in nii_files]
    results = p.starmap_async(load_remove_save, zip(input_files, out_files, [for_which_classes] * len(input_files),
                                                    [min_valid_object_size] * len(input_files)))
    res = results.get()
    p.close()
    p.join()
