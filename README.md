# Medical Coding with Modern Encoders and Modern Methods

This repository is **forked from [Explainable Medical Coding](https://github.com/JoakimEdin/explainable-medical-coding)**, which accompanied the paper  
[*An Unsupervised Approach to Achieve Supervised-Level Explainability in Healthcare Records* (Edin et al., EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.280/).

The original project focused on **explainability** in automated medical coding.  
This fork extends the codebase in a new direction: my work does **not focus on explainability**, although some explainability components remain in the repo due to its origins. These may be leveraged in **future research**.

---

## Project Focus

This project centers on developing and benchmarking new architectures for automated medical coding, including **PLM-RR (Pretrained Language Model with Retrieval and Re-ranking)**.  
The goal is to improve classification performance on large-scale clinical coding tasks by integrating retrieval-augmented methods into the training pipeline.  

---

## Setup

The setup largely follows the original repository. Here is a guide to setting up the repository for experimentation and reproducibility.
1. Clone this repository.
2. cd Medical-coding-in-2025
3. `cd data/raw`
4. Install [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) using wget (we used version 2.2)
5. Install [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/) using wget (we used version 2.2)
7. Back to the main repository folder `cd -`
8. Use a virtual environment (e.g., conda) with python 3.11.5 installed.
9. Create a weights and biases account. It is possible to run the experiments without wandb through using debug=true flag.
10. Prepare environment, dataset and models by running the commands 1. "make setup" 2. "make mimiciv" 3. "make download_roberta" 4. "make download_Modern_Neo"

You are now all set to run experiments!

# Note on licenses

## MIMIC
You need to obtain a non-commercial licence from physionet to use MIMIC. You will need to complete training. The training is free, but takes a couple of hours. - [link to data access](https://physionet.org/content/mimiciii/1.4/)

## Model weights can only be used non-commercially
While we would love to make everything fully open source, we cannot. Becaue MIMIC has a non-commercial license, the models trained using that data will also have a non-commercial licence. Therefore, using our models or RoBERTa-base-PM-M3-Voc's weights for commercial usecases is forbidden.

# How to run experiments
## How to train a model
You can run any experiment found in `explainable_medical_coding/configs/experiment`. Here are some examples:
   * Train PLM-ICD on MIMIC-III full and MDACE on GPU 0: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd gpu=0`
   * Train PLM-ICD using the supervised approach proposed by Cheng et al. on MIMIC-III full and MDACE on GPU 0: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd_supervised gpu=0`
   * Train PLM-ICD using input gradient regularization on MIMIC-III full and MDACE on GPU 0: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd_igr gpu=0`
   * Train PLM-ICD using token masking on MIMIC-III full and MDACE on GPU 0: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd_tm gpu=0`
   * Train PLM-ICD using projected gradient descent on MIMIC-III full and MDACE on GPU 0: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd_pgd gpu=0`
   * Train PLM-ICD on MIMIC-III full and MDACE on GPU 0 using a batch_size of 1: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd gpu=0 dataloader.max_batch_size=1`
   * Train PLM-ICD on MIMIC-IV ICD-10 and MDACE on GPU 0: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd gpu=0 dataloader.max_batch_size=1 data=mimiciv_icd10`

# Overview of the repository
#### configs
We use [Hydra](https://hydra.cc/docs/intro/) for configurations. The configs for every experiment is found in `explainable_medical_coding/configs/experiments`. Furthermore, the configuration for the sweeps are found in `explainable_medical_coding/configs/sweeps`. We used [Weights and Biases Sweeps](https://docs.wandb.ai/guides/sweeps) for most of our experiments.

#### data
This is where the splits and datasets are stored

#### models
The directory contains the model weights.

#### reports
This is the code used to generate the plots and tables used in the paper. The code uses the Weights and Biases API to fetch the experiment results. The code is not usable by others, but was included for the possibility to validate our figures and tables.

#### explainable_medical_coding
This is were the code for running the experiments and evaluating explanation methods are found.

# My setup
I ran the experiments on one A100 80GB per experiment. I had 2TB RAM on my machine.

# Other resources
Check out [my blog post](https://substack.com/home/post/p-145913061?source=queue) criticizing popular ideas in automated medical coding. Also, check out my blog post, [Gradients are not Explanations](https://substack.com/home/post/p-148104869?source=queue). I think it will be interesting for most researchers in the field.

# Acknowledgement
Thank you, Jonas Lyngsø, for providing the template for making the datasets in explainable_medical_coding/datasets/.
