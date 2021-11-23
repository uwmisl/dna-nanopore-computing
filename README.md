# DNA Circuit Detection from Raw Nanopore Sensing Data
Analysis pipeline for extracting, filtering, classifying, and quantifying DNA circuit output on a nanopore sensing platform. Bulk raw data was collected from Oxford Nanopore Technologies' MinION using R9.4.1 flow cells and a custom MinKNOW run script.

Adapted from https://github.com/uwmisl/NanoporeTERs.

## System Requirements and Installation
This software is compatible with Linux operating systems. The classification algorithms in this software also utilize a GPU (CUDA 10.0).

This repository was developed on a Jupyter notebook server running Python 2.7. The following dependencies should be installed.

Packages:
* dask (1.2.2)
* future (0.17.1)
* joblib (0.14.0)
* matplotlib (2.2.4)
* numpy (1.16.2)
* pandas (0.24.2)
* pyyaml (5.1.2)
* scikit-learn (0.20.4)
* scipy (1.2.2)
* torch (1.2.0)
* yaml (0.1.7)

## How to Use
Open `nanopore_experiments/prep_experiment_notebook.ipynb`. Change the date in Cell 2 to match the appropriate experiment and run the entire notebook. This will create a new experiment notebook in `nanopore_experiments` under the name `experiment_DATE_FLOWCELL.ipynb`, as well as a config file in `nanopore_experiments/configs` under the name `segment_DATE_FLOWCELL.yml`. 

Open the newly generated experiment notebook. Instructions are written in the notebook on the expected behavior and available parameters for each major step in the data processing pipeline. All cells in the notebook should be run in sequential order.

## Demo
TBD
