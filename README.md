# E2MISeg: Enhancing Edge-aware 3D Medical Image Segmentation via Feature Progressive Co-aggregation

3D segmentation is critically essential in the clinical medical field, which aids physicians in locating lesions and assists in clinical decision-making. The unique properties of organ and tumour images with large-scale variations and low-edge pixel-level contrast make clear segment edges difficult. Facing these problems, we propose an **Enhancing Edge-aware Medical Image Seg**mentation (E2MISeg) for smooth segmentation in boundary ambiguity. Firstly, we propose the Multi-level Feature Group Aggregation (MFGA) module to enhance the accuracy of edge voxel classification through the boundary clue of lesion tissue and background. Secondly, to minimize the influence of background noise on the model's sensitivity to the foreground, the Hybrid Feature Representation (HFR) block utilizes an interactive CNN and Transformer to deeply mine the lesion area and edge texture features while providing more clues for the MFGA module. Finally, we introduce the Scale-Sensitive (SS) loss function that dynamically adjusts the weights assigned to targets based on segmentation errors, with these weights guiding the network to focus on regions where segmentation edges are unclear. Furthermore, we retrospectively collated the Mantle Cell Lymphoma PET Imaging Diagnosis (MCLID) dataset of 176 patients from multiple central hospitals, which enhances our algorithm's robustness against complex clinical data. The extensive experimental results on three public challenge datasets and the MCLID clinical dataset demonstrate our approach, which outperforms the state-of-the-art methods. Further analysis shows that our components work together to achieve smooth edge segmentation, which is of great significance for accurate clinical diagnosis and prognosis analysis.

### **UPDATE**

- (8 15, 2025): update code and readme.

- (6 11, 2025): Related comparison model implementation reference: [MedNeXt] (https://github.com/MIC-DKFZ/MedNeXt/tree/main/nnunet_mednext/network_architecture/custom_modules/custom_networks) (sorry, I am doing internship)

- (5 16, 2025): Lung-related network architecture and trainer uploaded after sorting(sorry, I am doing internship)

- (8 15, 2024): upload  code.


<hr />

### **Installation**

```
1.sys requirement: Pytorch=2.0.1, CUDA=11.8
2.env Installation: conda env create -f environment.yaml
```

<hr />

### **Dataset**

**Dataset I**

The dataset folders for [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) should be organized as follows:
```
./DATASET_Acdc/
  ├── e2miseg_raw/
      ├── e2miseg_raw_data/
           ├── Task01_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
           ├── Task001_ACDC
       ├── e2miseg_cropped_data/
           ├── Task001_ACDC
```

**Dataset II**

The format of [Brain_tumor](http://medicaldecathlon.com/) dataset is the same as ACDC

**Dataset III** (MCL)

**Dataset IV**

The format of [Decathlon-lung](http://medicaldecathlon.com/) dataset is the same as ACDC

<hr />

### **Training**

1.ACDC
```
bash run_training_acdc.sh
```
2.BraTS
```
bash run_training_tumor.sh
```
3.Mcl
```
bash run_training_mcl.sh
```
4.Lung
```
bash run_training_lung.sh
```

<hr />

### **Evaluation**

1.ACDC
```
bash Acdc_run_predict.sh
```
2.BraTS
```
bash Tumor_run_predict.sh
```
3.Mcl
```
bash Mcl_run_predict.sh
```
4.Lung
```
bash lung_run_predict.sh
```
<hr />

### **Acknowledgement**

[nnUNet](https://github.com/MIC-DKFZ/nnUNet)、[nnFormer](https://github.com/282857341/nnFormer)、[MedNeXt](https://github.com/MIC-DKFZ/MedNeXt)、[UNETR++](https://github.com/Amshaker/unetr_plus_plus)

<hr />

### **Citation**
If you use our work, please consider citing:
```
@article{2026e2miseg,
  title={E2MISeg: Enhancing edge-aware 3D medical image segmentation via feature progressive co-aggregation},
  author={Jiang, Lincen and Xu, Wenpin and Zheng, Xinyuan and Zhang, Zitong and Jiang, Zekun and Jiang, Chong and Chen, Yanli and Ji, Yimu and Liu, Shangdong and Liu, Jianwei and others},
  journal={Expert Systems with Applications},
  volume={296},
  pages={128861},
  year={2026},
  publisher={Elsevier}
}
```

<hr />

### **Contact**

Should you have any question, please create an issue on this repository or contact me at sksmile.v@gmail.com.

<hr />
