<div align="center">
<h1 align="center">🫁 DRU-Net </h1>
<h3 align="center">Segmentation of Non-Small Cell Lung Carcinomas
</h3>

[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/AICAN-Research/DRU-Net/blob/main/LICENSE.md)
[![CI/CD](https://github.com/AICAN-Research/DRU-Net/actions/workflows/deploy.yml/badge.svg)](https://github.com/AICAN-Research/DRU-Net/actions/workflows/linting.yml)
<a target="_blank" href="https://huggingface.co/spaces/andreped/AeroPath"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow.svg"></a>
<a href="https://github.com/AICAN-Research/DRU-Net/blob/main/notebooks/TrainingOnWSIPatches.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
[![arXiv preprint](https://img.shields.io/badge/arXiv-preprint-D12424)](
https://doi.org/10.48550/arXiv.2406.14287)

**DRU-Net** was developed the AICAN research group.

</div>

# Introduction

This repository contains the source code related to the manuscript _"DRU-Net: Lung carcinoma segmentation using multi-lens distortion and fusion refinement network"_ which is openly available on [arXiv](https://arxiv.org/abs/2406.14287).

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


<details>
<summary>

# Development</summary>

## Linting

First install linting dependencies:

```
pip install black==22.3.0 isort==5.10.1 flake8==4.0.1
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
@misc{oskouei2024segmentation,
    title={{Segmentation of Non-Small Cell Lung Carcinomas: Introducing DRU-Net and Multi-Lens Distortion}}, 
    author={Soroush Oskouei and Marit Valla and André Pedersen and Erik Smistad and Vibeke Grotnes Dale and Maren Høibø and Sissel Gyrid Freim Wahl and Mats Dehli Haugum and Thomas Langø and Maria Paula Ramnefjell and Lars Andreas Akslen and Gabriel Kiss and Hanne Sorger},
    year={2024},
    eprint={2406.14287},
    archivePrefix={arXiv},
    doi={10.48550/arXiv.2406.14287}
}
```