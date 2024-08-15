import shutil
from collections import OrderedDict
from copy import deepcopy
import e2miseg
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from e2miseg.run.default_configuration import default_num_threads
from e2miseg.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from e2miseg.experiment_planning.common_utils import get_pool_and_conv_props_poolLateV2
from e2miseg.experiment_planning.utils import create_lists_from_splitted_dataset
from e2miseg.network_architecture.generic_UNet import Generic_UNet
from e2miseg.paths import *
from e2miseg.preprocessing.cropping import get_case_identifier_from_npz
from e2miseg.training.model_restore import recursive_find_python_class


class ExperimentPlanner(object):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        self.folder_with_cropped_data = folder_with_cropped_data
        self.preprocessed_output_folder = preprocessed_output_folder
        self.list_of_cropped_npz_files = subfiles(self.folder_with_cropped_data, True, None, ".npz", True)

        self.preprocessor_name = "GenericPreprocessor"

        assert isfile(join(self.folder_with_cropped_data, "dataset_properties.pkl")), \
            "folder_with_cropped_data must contain dataset_properties.pkl"
        self.dataset_properties = load_pickle(join(self.folder_with_cropped_data, "dataset_properties.pkl"))

        self.plans_per_stage = OrderedDict()
        self.plans = OrderedDict()
        self.plans_fname = join(self.preprocessed_output_folder, "nnFormerPlans" + "fixed_plans_3D.pkl")
        self.data_identifier = default_data_identifier

        self.transpose_forward = [0, 1, 2]
        self.transpose_backward = [0, 1, 2]

        self.unet_base_num_features = Generic_UNet.BASE_NUM_FEATURES_3D  # 30
        self.unet_max_num_filters = 320
        self.unet_max_numpool = 999
        self.unet_min_batch_size = 2
        self.unet_featuremap_min_edge_length = 4

        self.target_spacing_percentile = 50
        self.anisotropy_threshold = 3
        self.how_much_of_a_patient_must_the_network_see_at_stage0 = 4
        self.batch_size_covers_max_percent_of_dataset = 0.05

        self.conv_per_stage = 2

    def get_target_spacing(self):
        spacings = self.dataset_properties['all_spacings']

        # target = np.median(np.vstack(spacings), 0)
        # if target spacing is very anisotropic we may want to not downsample the axis with the worst spacing
        # uncomment after mystery task submission
        """worst_spacing_axis = np.argmax(target)
        if max(target) > (2.5 * min(target)):
            spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
            target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 5)
            target[worst_spacing_axis] = target_spacing_of_that_axis"""

        target = np.percentile(np.vstack(spacings), self.target_spacing_percentile, 0)
        return target

    def save_my_plans(self):
        with open(self.plans_fname, 'wb') as f:
            pickle.dump(self.plans, f)

    def load_my_plans(self):
        self.plans = load_pickle(self.plans_fname)

        self.plans_per_stage = self.plans['plans_per_stage']
        self.dataset_properties = self.plans['dataset_properties']

        self.transpose_forward = self.plans['transpose_forward']
        self.transpose_backward = self.plans['transpose_backward']

    def determine_postprocessing(self):
        pass


    def get_properties_for_stage(self, current_spacing, original_spacing, original_shape, num_cases,
                                 num_modalities, num_classes):

        new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)
        dataset_num_voxels = np.prod(new_median_shape) * num_cases


        input_patch_size = 1 / np.array(current_spacing)


        input_patch_size /= input_patch_size.mean()


        input_patch_size *= 1 / min(input_patch_size) * 512
        input_patch_size = np.round(input_patch_size).astype(int)

        input_patch_size = [min(i, j) for i, j in zip(input_patch_size, new_median_shape)]

        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
        shape_must_be_divisible_by = get_pool_and_conv_props_poolLateV2(input_patch_size,
                                                                        self.unet_featuremap_min_edge_length,
                                                                        self.unet_max_numpool,
                                                                        current_spacing)

        ref = Generic_UNet.use_this_for_batch_size_computation_3D
        here = Generic_UNet.compute_approx_vram_consumption(new_shp, network_num_pool_per_axis,
                                                            self.unet_base_num_features,
                                                            self.unet_max_num_filters, num_modalities,
                                                            num_classes,
                                                            pool_op_kernel_sizes, conv_per_stage=self.conv_per_stage)
        while here > ref:
            axis_to_be_reduced = np.argsort(new_shp / new_median_shape)[-1]

            tmp = deepcopy(new_shp)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by_new = \
                get_pool_and_conv_props_poolLateV2(tmp,
                                                   self.unet_featuremap_min_edge_length,
                                                   self.unet_max_numpool,
                                                   current_spacing)
            new_shp[axis_to_be_reduced] -= shape_must_be_divisible_by_new[axis_to_be_reduced]

            # we have to recompute numpool now:
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
            shape_must_be_divisible_by = get_pool_and_conv_props_poolLateV2(new_shp,
                                                                            self.unet_featuremap_min_edge_length,
                                                                            self.unet_max_numpool,
                                                                            current_spacing)

            here = Generic_UNet.compute_approx_vram_consumption(new_shp, network_num_pool_per_axis,
                                                                self.unet_base_num_features,
                                                                self.unet_max_num_filters, num_modalities,
                                                                num_classes, pool_op_kernel_sizes,
                                                                conv_per_stage=self.conv_per_stage)
            # print(new_shp)

        input_patch_size = new_shp

        batch_size = Generic_UNet.DEFAULT_BATCH_SIZE_3D  # This is what works with 128**3
        batch_size = int(np.floor(max(ref / here, 1) * batch_size))

        # check if batch size is too large
        max_batch_size = np.round(self.batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
                                  np.prod(input_patch_size, dtype=np.int64)).astype(int)
        max_batch_size = max(max_batch_size, self.unet_min_batch_size)
        batch_size = max(1, min(batch_size, max_batch_size))

        do_dummy_2D_data_aug = (max(input_patch_size) / input_patch_size[
            0]) > self.anisotropy_threshold

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_num_pool_per_axis,
            'patch_size': input_patch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'do_dummy_2D_data_aug': do_dummy_2D_data_aug,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
        }
        return plan

    def plan_experiment(self):
        use_nonzero_mask_for_normalization = self.determine_whether_to_use_mask_for_norm()
        print("Are we using the nonzero mask for normalizaion?", use_nonzero_mask_for_normalization)
        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']

        all_classes = self.dataset_properties['all_classes']
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        target_spacing = self.get_target_spacing()
        new_shapes = [np.array(i) / target_spacing * np.array(j) for i, j in zip(spacings, sizes)]

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
        self.transpose_forward = [max_spacing_axis] + remaining_axes
        self.transpose_backward = [np.argwhere(np.array(self.transpose_forward) == i)[0][0] for i in range(3)]

        # we base our calculations on the median shape of the datasets
        median_shape = np.median(np.vstack(new_shapes), 0)
        print("the median shape of the dataset is ", median_shape)

        max_shape = np.max(np.vstack(new_shapes), 0)
        print("the max shape in the dataset is ", max_shape)
        min_shape = np.min(np.vstack(new_shapes), 0)
        print("the min shape in the dataset is ", min_shape)

        print("we don't want feature maps smaller than ", self.unet_featuremap_min_edge_length, " in the bottleneck")

        self.plans_per_stage = list()

        target_spacing_transposed = np.array(target_spacing)[self.transpose_forward]
        median_shape_transposed = np.array(median_shape)[self.transpose_forward]
        print("the transposed median shape of the dataset is ", median_shape_transposed)

        print("generating configuration for 3d_fullres")
        self.plans_per_stage.append(self.get_properties_for_stage(target_spacing_transposed, target_spacing_transposed,
                                                                  median_shape_transposed,
                                                                  len(self.list_of_cropped_npz_files),
                                                                  num_modalities, len(all_classes) + 1))


        architecture_input_voxels_here = np.prod(self.plans_per_stage[-1]['patch_size'], dtype=np.int64)
        if np.prod(median_shape) / architecture_input_voxels_here < \
                self.how_much_of_a_patient_must_the_network_see_at_stage0:
            more = False
        else:
            more = True

        if more:
            print("generating configuration for 3d_lowres")

            lowres_stage_spacing = deepcopy(target_spacing)
            num_voxels = np.prod(median_shape, dtype=np.float64)
            while num_voxels > self.how_much_of_a_patient_must_the_network_see_at_stage0 * architecture_input_voxels_here:
                max_spacing = max(lowres_stage_spacing)
                if np.any((max_spacing / lowres_stage_spacing) > 2):
                    lowres_stage_spacing[(max_spacing / lowres_stage_spacing) > 2] \
                        *= 1.01
                else:
                    lowres_stage_spacing *= 1.01
                num_voxels = np.prod(target_spacing / lowres_stage_spacing * median_shape, dtype=np.float64)

                lowres_stage_spacing_transposed = np.array(lowres_stage_spacing)[self.transpose_forward]
                new = self.get_properties_for_stage(lowres_stage_spacing_transposed, target_spacing_transposed,
                                                    median_shape_transposed,
                                                    len(self.list_of_cropped_npz_files),
                                                    num_modalities, len(all_classes) + 1)
                architecture_input_voxels_here = np.prod(new['patch_size'], dtype=np.int64)
            if 2 * np.prod(new['median_patient_size_in_voxels'], dtype=np.int64) < np.prod(
                    self.plans_per_stage[0]['median_patient_size_in_voxels'], dtype=np.int64):
                self.plans_per_stage.append(new)

        self.plans_per_stage = self.plans_per_stage[::-1]
        self.plans_per_stage = {i: self.plans_per_stage[i] for i in range(len(self.plans_per_stage))}  # convert to dict

        print(self.plans_per_stage)
        print("transpose forward", self.transpose_forward)
        print("transpose backward", self.transpose_backward)

        normalization_schemes = self.determine_normalization_scheme()
        only_keep_largest_connected_component, min_size_per_class, min_region_size_per_class = None, None, None
        # removed training data based postprocessing. This is deprecated

        # these are independent of the stage
        plans = {'num_stages': len(list(self.plans_per_stage.keys())), 'num_modalities': num_modalities,
                 'modalities': modalities, 'normalization_schemes': normalization_schemes,
                 'dataset_properties': self.dataset_properties, 'list_of_npz_files': self.list_of_cropped_npz_files,
                 'original_spacings': spacings, 'original_sizes': sizes,
                 'preprocessed_data_folder': self.preprocessed_output_folder, 'num_classes': len(all_classes),
                 'all_classes': all_classes, 'base_num_features': self.unet_base_num_features,
                 'use_mask_for_norm': use_nonzero_mask_for_normalization,
                 'keep_only_largest_region': only_keep_largest_connected_component,
                 'min_region_size_per_class': min_region_size_per_class, 'min_size_per_class': min_size_per_class,
                 'transpose_forward': self.transpose_forward, 'transpose_backward': self.transpose_backward,
                 'data_identifier': self.data_identifier, 'plans_per_stage': self.plans_per_stage,
                 'preprocessor_name': self.preprocessor_name,
                 'conv_per_stage': self.conv_per_stage,
                 }

        self.plans = plans
        self.save_my_plans()

    def determine_normalization_scheme(self):
        schemes = OrderedDict()
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        for i in range(num_modalities):
            if modalities[i] == "CT" or modalities[i] == 'ct':
                schemes[i] = "CT"
            else:
                schemes[i] = "nonCT"
        return schemes

    def save_properties_of_cropped(self, case_identifier, properties):
        with open(join(self.folder_with_cropped_data, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)

    def load_properties_of_cropped(self, case_identifier):
        with open(join(self.folder_with_cropped_data, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return properties

    def determine_whether_to_use_mask_for_norm(self):
        # only use the nonzero mask for normalization of the cropping based on it resulted in a decrease in
        # image size (this is an indication that the data is something like brats/isles and then we want to
        # normalize in the brain region only)
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))
        use_nonzero_mask_for_norm = OrderedDict()

        for i in range(num_modalities):
            if "CT" in modalities[i]:
                use_nonzero_mask_for_norm[i] = False
            else:
                all_size_reductions = []
                for k in self.dataset_properties['size_reductions'].keys():
                    all_size_reductions.append(self.dataset_properties['size_reductions'][k])

                if np.median(all_size_reductions) < 3 / 4.:
                    print("using nonzero mask for normalization")
                    use_nonzero_mask_for_norm[i] = True
                else:
                    print("not using nonzero mask for normalization")
                    use_nonzero_mask_for_norm[i] = False

        for c in self.list_of_cropped_npz_files:
            case_identifier = get_case_identifier_from_npz(c)
            properties = self.load_properties_of_cropped(case_identifier)
            properties['use_nonzero_mask_for_norm'] = use_nonzero_mask_for_norm
            self.save_properties_of_cropped(case_identifier, properties)
        use_nonzero_mask_for_normalization = use_nonzero_mask_for_norm
        return use_nonzero_mask_for_normalization

    def write_normalization_scheme_to_patients(self):
        for c in self.list_of_cropped_npz_files:
            case_identifier = get_case_identifier_from_npz(c)
            properties = self.load_properties_of_cropped(case_identifier)
            properties['use_nonzero_mask_for_norm'] = self.plans['use_mask_for_norm']
            self.save_properties_of_cropped(case_identifier, properties)

    def run_preprocessing(self, num_threads):
        if os.path.isdir(join(self.preprocessed_output_folder, "gt_segmentations")):
            shutil.rmtree(join(self.preprocessed_output_folder, "gt_segmentations"))
        shutil.copytree(join(self.folder_with_cropped_data, "gt_segmentations"),
                        join(self.preprocessed_output_folder, "gt_segmentations"))
        normalization_schemes = self.plans['normalization_schemes']
        use_nonzero_mask_for_normalization = self.plans['use_mask_for_norm']
        intensityproperties = self.plans['dataset_properties']['intensityproperties']
        preprocessor_class = recursive_find_python_class([join(e2miseg.__path__[0], "preprocessing")],
                                                         self.preprocessor_name, current_module="e2miseg.preprocessing")
        assert preprocessor_class is not None
        preprocessor = preprocessor_class(normalization_schemes, use_nonzero_mask_for_normalization,
                                         self.transpose_forward,
                                          intensityproperties)
        target_spacings = [i["current_spacing"] for i in self.plans_per_stage.values()]
        if self.plans['num_stages'] > 1 and not isinstance(num_threads, (list, tuple)):
            num_threads = (default_num_threads, num_threads)
        elif self.plans['num_stages'] == 1 and isinstance(num_threads, (list, tuple)):
            num_threads = num_threads[-1]
        preprocessor.run(target_spacings, self.folder_with_cropped_data, self.preprocessed_output_folder,
                         self.plans['data_identifier'], num_threads)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_ids", nargs="+", help="list of int")
    parser.add_argument("-p", action="store_true", help="set this if you actually want to run the preprocessing. If "
                                                        "this is not set then this script will only create the plans file")
    parser.add_argument("-tl", type=int, required=False, default=4, help="num_threads_lowres")
    parser.add_argument("-tf", type=int, required=False, default=4, help="num_threads_fullres")

    args = parser.parse_args()
    task_ids = args.task_ids
    run_preprocessing = args.p
    tl = args.tl
    tf = args.tf

    tasks = []
    for i in task_ids:
        i = int(i)
        candidates = subdirs(nnFormer_cropped_data, prefix="Task%03.0d" % i, join=False)
        assert len(candidates) == 1
        tasks.append(candidates[0])

    for t in tasks:
        try:
            print("\n\n\n", t)
            cropped_out_dir = os.path.join(nnFormer_cropped_data, t)
            preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)
            splitted_4d_output_dir_task = os.path.join(nnFormer_raw_data, t)
            lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

            dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False)
            _ = dataset_analyzer.analyze_dataset()

            maybe_mkdir_p(preprocessing_output_dir_this_task)
            shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
            shutil.copy(join(nnFormer_raw_data, t, "dataset.json"), preprocessing_output_dir_this_task)

            threads = (tl, tf)

            print("number of threads: ", threads, "\n")

            exp_planner = ExperimentPlanner(cropped_out_dir, preprocessing_output_dir_this_task)
            exp_planner.plan_experiment()
            if run_preprocessing:
                exp_planner.run_preprocessing(threads)
        except Exception as e:
            print(e)
