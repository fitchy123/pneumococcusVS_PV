### Repository for "Artificial intelligence-guided phenotypic virtual screening against drug-resistant Streptococcus Pneumoniae"

A study using AI to find potential antibiotics effective against drug-resistant *Streptococcus pneumoniae*.

This dataset contains code to run screening of the drug repurposing hub with pre-trained models.

#### Repository Structure
This repository contains:
- Pre-processed dataset used in model training (processed_datasets/10uM_FP_clustered_resistant_pneumococcus_augmented_dataset.csv)
- Drug repurposing hub screening dataset (https://repo-hub.broadinstitute.org/repurposing)
- Instructions for obtaining saved molformer-based models (README.md)
- Code for evaluating models (models/molformer)
- Instructions for running evaluation (README.md)
- Files for ensuring the correct packages are installed (environment.yml and requirements.txt)

#### Running models

The saved molformer models can be downloaded from zenodo: https://zenodo.org/records/14960323

These models should be downloaded and saved in the repository, by default the paths are set to look for them in the "model_checkpoints" folder but you can also pass your own path as a command line argument. 

File to run virtual screening:
- models/molformer/lightning_predict.py
File to analyse results:
- results/results_analysis.py

#### Package Installation
- Set up conda environment with environment.yml: `conda env create -f environment.yml`
- Install pip packages: `pip install -r requirements.txt`
