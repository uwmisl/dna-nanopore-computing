# DNA Nanopore Computing
Analysis pipeline for extracting, filtering, classifying, and quantifying DNA circuit output on a nanopore sensing platform (Oxford Nanopore Technologies).

Adapted from https://github.com/uwmisl/NanoporeTERs.

## Installation and Dependencies
This software is compatible with Linux operating systems and has been tested on CentOS 7.9.

Clone repository and install the following dependencies.

Environment:
* Jupyter notebook running Python 2.7.16

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
