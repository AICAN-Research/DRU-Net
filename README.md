<div align="center">
<h1 align="center">🫁 DRU-Net </h1>
<h3 align="center">Segmentation of Non-Small Cell Lung Carcinomas
</h3>

[![license](https://img.shields.io/badge/MIT-License-008000)](https://github.com/AICAN-Research/DRU-Net/blob/main/LICENSE.md)
<a href="https://github.com/AICAN-Research/DRU-Net/blob/main/notebooks/TrainingOnWSIPatches.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
[![arXiv preprint](https://img.shields.io/badge/arXiv-preprint-D12424)](
https://doi.org/10.48550/arXiv.2406.14287)

**DRU-Net** was developed by the AICAN research group.

</div>

# Introduction

This repository contains the source code related to the manuscript _"DRU-Net: Lung carcinoma segmentation using multi-lens distortion and fusion refinement network"_ which is openly available on [arXiv](https://arxiv.org/abs/2406.14287).

## A brief introduction to the DRU-Net model
The DRU-Net is an innovative artificial intelligence model designed for the segmentation of non-small cell lung carcinomas (NSCLCs) from whole slide images (WSIs). This model enhances the automated analysis of histopathological images, assisting pathologists by improving the speed and accuracy of cancer diagnosis.

## DRU-Net Architecture
DRU-Net integrates two main components:
1. **Patch-Wise Classification (PWC):** Utilizes truncated, pre-trained DenseNet201 and ResNet101V2 networks to classify image patches as tumor or non-tumor. This dual-head setup captures local features effectively, improving initial classification accuracy.
2. **Refinement Network:** A lightweight U-Net architecture that refines the segmentation by considering the global context of the WSIs. This network processes the concatenated outputs of the PWC and a down-sampled WSI, enhancing the detail and accuracy of the tumor delineation.

The model is trained on a proprietary dataset of annotated NSCLC WSIs, achieving a high Dice similarity coefficient of 0.91, indicating robust segmentation performance.

## Multi-lens Distortion Augmentation
To further enhance the model's robustness and accuracy, the DRU-Net employs a novel data augmentation technique called Multi-lens Distortion. This method simulates random local lens distortions (both barrel and pincushion) within the training images, introducing variability that helps the model generalize better to new, unseen data.

### How It Works:
- **Random Distortions:** Multiple "lenses" are randomly placed on the training images, each distorting the image within a specific radius and strength.
- **Implementation:** The algorithm adjusts pixel positions based on the distortion parameters, creating variations in cell and tissue shapes that the model might encounter in real pathological slides.

In the related research, this augmentation resulted in improving the network's performance by 3% on tasks requiring high generalization capabilities.

# Getting started

## Setup

1. Setup virtual environment and activate it:

```
python -m venv venv/
source venv/bin/activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```


# Development

## Automatic training with whole tumor annotation (WTA)
For this approach, after creating thumbnails of your WSIs, you need to create tissue clusters for balancing the data between various tissue types. For implementation details, see [ClusteringForTissueBalancing.py](ClusteringForTissueBalancing.py).
You also need to create gradients which will be used to create a reliable tissue mask and for clustering in later codes [GenerateGradients.py](GenerateGradients.py).
Before training, we also need to create positions of the patches in WSIs. To generate related patches and their labels you can use npy files or rgb image formats and save the annotation mask. In our code, annotation masks are saved as images and are used for label extraction linked with the patch position. For implementation see [PatchPositions.ipynb](notebooks/PatchPositions.ipynb).
Finally, you can train your model using a code similar to [TrainingOnWSIPatches.ipynb](notebooks/TrainingOnWSIPatches.ipynb).
A sample data generator is used here [Generator.py](src/Generator.py).

## Training using saved patches with partial selective annotation (PSA)
In this method, you need to save images from areas of interest and then train the model on those images. Sample implementation can be found here [ManyShotTransferLearning.ipynb](notebooks/ManyShotTransferLearning.ipynb).
You can also use a few-shot learning method [FewShot.py](./FewShot.py) and optimize for the number of classes using our novel optimization method using [FindOptimumNumberOfClasses.py](FindOptimumNumberOfClasses.py).


<details>
<summary>

## Linting</summary>

First install linting dependencies:

```
pip install isort==5.10.1 flake8==4.0.1 black==22.3.0 "black[jupyter]"
```

Then run linting test by:

```
sh shell/lint.sh
```

Perform automatic linting by:

```
sh shell/format.sh
```

</details>


## Citation

If you found the source code or manuscript relevant in your research, please cite the following reference:

```
@Article{jimaging11050166,
AUTHOR = {Oskouei, Soroush and Valla, Marit and Pedersen, André and Smistad, Erik and Dale, Vibeke Grotnes and Høibø, Maren and Wahl, Sissel Gyrid Freim and Haugum, Mats Dehli and Langø, Thomas and Ramnefjell, Maria Paula and Akslen, Lars Andreas and Kiss, Gabriel and Sorger, Hanne},
TITLE = {Segmentation of Non-Small Cell Lung Carcinomas: Introducing DRU-Net and Multi-Lens Distortion},
JOURNAL = {Journal of Imaging},
VOLUME = {11},
YEAR = {2025},
NUMBER = {5},
ARTICLE-NUMBER = {166},
URL = {https://www.mdpi.com/2313-433X/11/5/166},
ISSN = {2313-433X},
ABSTRACT = {The increased workload in pathology laboratories today means automated tools such as artificial intelligence models can be useful, helping pathologists with their tasks. In this paper, we propose a segmentation model (DRU-Net) that can provide a delineation of human non-small cell lung carcinomas and an augmentation method that can improve classification results. The proposed model is a fused combination of truncated pre-trained DenseNet201 and ResNet101V2 as a patch-wise classifier, followed by a lightweight U-Net as a refinement model. Two datasets (Norwegian Lung Cancer Biobank and Haukeland University Lung Cancer cohort) were used to develop the model. The DRU-Net model achieved an average of 0.91 Dice similarity coefficient. The proposed spatial augmentation method (multi-lens distortion) improved the Dice similarity coefficient from 0.88 to 0.91. Our findings show that selecting image patches that specifically include regions of interest leads to better results for the patch-wise classifier compared to other sampling methods. A qualitative analysis by pathology experts showed that the DRU-Net model was generally successful in tumor detection. Results in the test set showed some areas of false-positive and false-negative segmentation in the periphery, particularly in tumors with inflammatory and reactive changes. In summary, the presented DRU-Net model demonstrated the best performance on the segmentation task, and the proposed augmentation technique proved to improve the results.},
DOI = {10.3390/jimaging11050166}
}
```
