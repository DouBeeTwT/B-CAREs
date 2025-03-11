# B-CAREs

## Early Risk Screening and Hierarchical Management of Breast Cancer Based on Artificial Intelligence Systems

## Abstract
Early screening for breast cancer using ultrasound has been shown to significantly improve patient prognosis. However, the widespread adoption of this method in large populations is limited by challenges in detection accuracy and the absence of standardized protocols. This study introduces the Breast Cancer Assessment, Reporting & Early-intervention System (B-CAREs) based on AI, which integrates advanced artificial intelligence algorithms, including large language models. The system is designed to automatically generate initial diagnostic reports. These reports encompass tumor identification, malignancy classification, histological subtype categorization, lymph node metastasis probability assessment, high-risk factor analysis, and treatment recommendations. We retrospectively trained and validated our model using data from 159,649 breast disease patients across seven clinical centers. The model achieved an average AUROC of 0.923 (95% CI 0.912–0.934), sensitivity of 0.906 (95% CI 0.895–0.917), and specificity of 0.907 (95% CI 0.898–0.916) on an independent test set, demonstrating strong multi-task prediction performance. Furthermore, 50 clinical experts assessed the reports based on guideline adherence and diagnostic accuracy, yielding an average score of 8.90 ± 0.25 out of 10. We subsequently conducted a prospective, double-blind validation study involving 1,229 patients from three independent multicenter cohorts. The results showed an average AUROC of 0.905 (95% CI 0.898–0.913), sensitivity of 0.889 (95% CI 0.880–0.898), and specificity of 0.890 (95% CI 0.881–0.899). The B-CAREs effectively increased the early screening detection rate by 25.4% and reduced the rate of missed and misdiagnoses by 19.4%. The B-CAREs system, through the implementation of an integrated "screening-stratification-interpretation" management workflow, enhances resource allocation efficiency in early-stage diagnosis and optimizes clinical decision-making capabilities, while concurrently advancing in-depth mechanistic understanding of disease pathogenesis.

For details, see [Paper]
![Figure1](scripts/Figures/Figure1.png)

This repository contains:

1. This is a code for the work being submitted, we provide only a brief description
2. This includes model structure, training code and part of test data

## Model architecture
![Figure4](scripts/Figures/Figure4.png)