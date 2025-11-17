### Repository for "Artificial intelligence-guided phenotypic virtual screening against drug-resistant Streptococcus Pneumoniae"

A study using AI to find potential antibiotics effective against drug-resistant *Streptococcus pneumoniae*.

This dataset contains code to run screening of the drug repurposing hub with pre-trained models.

#### Repository Structure
This repository contains:
- Pre-processed dataset used in model training- processed_datasets/10uM_FP_clustered_resistant_pneumococcus_augmented_dataset.csv
- Drug repurposing hub screening dataset- processed_datasets/repurposing_samples.csv (https://repo-hub.broadinstitute.org/repurposing)
- Instructions for obtaining saved models (README.md)
- Code for evaluating models (models/)
- Instructions for running evaluation (README.md)
- Files for ensuring the correct packages are installed (environment.yml and requirements.txt)

#### Virtual Screening Results
To analyse the results of ML methods on the drug repurposing hub run:
- results_analysis.py

#### Running models

The saved models can be downloaded from zenodo: https://zenodo.org/records/14960323\
Within each model folder there is a hparams.txt file listing hyperparameters of the saved models for reference (saved models from zenodo will run on default code with no changes necessary).

These models should be downloaded and saved in the repository, by default the paths are set to look for them in the "model_checkpoints" folder but you can also pass your own path as a command line argument. 

Files to run virtual screening:
- MolFormer: models/molformer/lightning_predict.py
- Chemprop: models/chemprop/run_chemprop.py
- Random Forest: models/random_forest/run_rf.py
Running these files will save the results of that ML method in a CSV file in the results directory

#### Package Installation
- Set up conda environment with environment.yml: `conda env create -f environment.yml`
- Install pip packages: `pip install -r requirements.txt`
