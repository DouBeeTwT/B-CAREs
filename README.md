# B-CAREs

## Early Risk Screening and Hierarchical Management of Breast Cancer Based on Artificial Intelligence Systems

## Abstract
Early screening for breast cancer using ultrasound has been shown to significantly improve patient prognosis. However, the widespread adoption of this method in large populations is limited by challenges in detection accuracy and the absence of standardized protocols. This study introduces the Breast Cancer Assessment, Reporting & Early-intervention System (B-CAREs) based on AI, which integrates advanced artificial intelligence algorithms, including large language models. The system is designed to automatically generate initial diagnostic reports. These reports encompass tumor identification, malignancy classification, histological subtype categorization, lymph node metastasis probability assessment, high-risk factor analysis, and treatment recommendations. We retrospectively trained and validated our model using data from 159,649 breast disease patients across seven clinical centers. The model achieved an average AUROC of 0.923 (95% CI 0.912–0.934), sensitivity of 0.906 (95% CI 0.895–0.917), and specificity of 0.907 (95% CI 0.898–0.916) on an independent test set, demonstrating strong multi-task prediction performance. Furthermore, 50 clinical experts assessed the reports based on guideline adherence and diagnostic accuracy, yielding an average score of 8.90 ± 0.25 out of 10. We subsequently conducted a prospective, double-blind validation study involving 1,229 patients from three independent multicenter cohorts. The results showed an average AUROC of 0.905 (95% CI 0.898–0.913), sensitivity of 0.889 (95% CI 0.880–0.898), and specificity of 0.890 (95% CI 0.881–0.899). The B-CAREs effectively increased the early screening detection rate by 25.4% and reduced the rate of missed and misdiagnoses by 19.4%. The B-CAREs system, through the implementation of an integrated "screening-stratification-interpretation" management workflow, enhances resource allocation efficiency in early-stage diagnosis and optimizes clinical decision-making capabilities, while concurrently advancing in-depth mechanistic understanding of disease pathogenesis.

For details, see [Paper]
![Figure1](https://github.com/DouBeeTwT/B-CAREs/blob/main/scripts/Figures/Figure1.png)

This repository contains:

1. This is a code for the work being submitted, we provide only a brief description
2. This includes model structure, training code and part of test data

## Model architecture
![Figure4](https://github.com/DouBeeTwT/B-CAREs/blob/main/scripts/Figures/Figure4.png)

## Install
This project uses requirements.txt.
```bash
$ pip install -r requirements.txt
```

## Datasets
We have shared part of the thyroid ultrasound dataset for verification. Please refer to this article if you use them. Please read the following information for data usage permissions and the conditions for accessing the full dataset.

- All data that fueled the findings can be found within the article and the Supplementary Information. The breast cancer datasets trained and analyzed during this study are available in a deidentified form to protect patient privacy. The minimum breast cancer dataset required to interpret, verify, and extend the findings of this study has been deposited.
    - Pre-processed imaging data (ultrasound images with anonymized metadata).
    - Clinical feature tables (age, gender, tumor size) with all direct identifiers removed.
- Due to ethical restrictions and patient confidentiality agreements, the full dataset (e.g., raw imaging data, detailed clinical records) cannot be made publicly available. This pertains to detailed clinical records and high-resolution imaging data that, even after de-identification, may pose a risk of re-identification given the unique characteristics of breast cancer cases. Researchers who wish to access additional data for non-commercial academic purposes may submit a formal request to the corresponding author. Requests will be reviewed by the institutional ethics committee and data custodians. The following conditions apply:
    - Purpose: Data will only be shared for research purposes that align with the original study objectives. 
    - Access Restrictions: Requesters must sign a data use agreement prohibiting re-identification or redistribution.
    - Data Retention: Approved data will be available for 2 years from the date of publication.

This dataset contains 100 breast ultrasound images, categorized into seven subtypes of breast carcinoma:

| Subtype                                   | Image Number | Label Number|
| :---------------------------------------- | -----------: | ----------: |
| Benign Fibroadenoma (BF)                  |          190 |         190 |
| Ductal Carcinoma In Situ (DCIS)           |           20 |          20 |
| Lobular Carcinoma In Situ (LCIS)          |           10 |          10 |
| Invasive Ductal Carcinoma (IDC)           |           60 |          60 |
| Invasive Lobular Carcinoma (ILC)          |           40 |          40 |
| Invasive Breast Cancer Non-special (IBCN) |           70 |          70 |
| Invasive Breast Cancer Undefined (IBCU)   |           40 |          40 |
| Sum                                       |          430 |         430 |

More labels(BI-RADS, lymph node) could be found in dataset/baseline.csv and the infomation structure like below:


| PictureName | BIRADS | Lymph |  Age  | Gender | Size | FamilyHistory | Menophania | Period | Duration |Pregnant | Menopause | Smoke | Drink |
| :---------- | -----: | ----: | ----: | -----: | ---: | ------------: | ---------: | -----: | -------: |-------: | --------: | ----: | ----: |
| BIC_0009    |      3 |     0 |    81 | Female | 17.8 |             0 |         12 |     33 |        4 |      No |       Yes |    No |    No |

## Model Weight

Download model .pth files by your self

| Task Name | Download to |Link |
| :---------| :----- |:----|
| BI-RADS   | pth/BIRADS  | https://pan.sjtu.edu.cn/web/share/ef3c18cf0557a22a5d65b35fca88e442 |
| Subtype   | pth/Subtype | https://pan.sjtu.edu.cn/web/share/34371324234abf37f70ff9b4e43a1f99 |
| Lymph     | pth/Lymph   | https://pan.sjtu.edu.cn/web/share/12b42e150ee920db0388446254c7342b |

## Usage
1. train models
```python
python train.py --root {YOUR DATA PATHWAY} --device cuda --batch_size 32 --learning_rate 1e-3 --epoch_max 150 --epoch_qp 120 \
                --class_num {YOUR CLASS NUMBER} --backbone_name StarNet --protonet_name DeepLabV3 \
                --loss_weight [0.8,1.0,1.0,1.0,1.0] --base_num 4 --score_threshold 0.05 --nms_threshold 0.6 --max_object_num 7 \
                --seed 1
# For more details of parameters， please use
python train.py --help
```
For training step, you may use several minutes to several hours depend on your dataset size and GPU.

2. test models
```python
python report.py
```
For testing step, the demo shows a single patient report just with around one minute.

## Citing
If you use our code and any information in your research, please consider citing with the following BibTex.
