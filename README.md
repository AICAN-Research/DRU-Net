# DRU-Net

## Introduction

This project presents DRU-Net: Lung carcinoma segmentation using multi-lens distortion and fusion refinement network.

## Getting started

### Setup

1. Setup virtual environment and activate it:

```
python -m venv venv/
source venv/bin/activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```

### Linting

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
