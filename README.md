# Improving Automatic ICD-10 Coding with Long-Context Transformers and Description-Aware Architectures

This repository is **forked from [Explainable Medical Coding](https://github.com/JoakimEdin/explainable-medical-coding)**, which accompanied the paper  
[*An Unsupervised Approach to Achieve Supervised-Level Explainability in Healthcare Records* (Edin et al., EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.280/).

The original project focused on **explainability** in automated medical coding.  
This fork extends the codebase in a new direction: my work does **not focus on explainability**, although some explainability components remain in the repo due to its origins. These may be leveraged in **future research**.

---

## Project Focus

This project investigates whether **modern encoders** and **external knowledge integration** can improve the state-of-the-art **PLM-CA** framework for **automatic ICD-10 coding** on **MIMIC-IV**. We:

- **Benchmark long-context encoders** (ModernBERT, BioClinical-ModernBERT, NeoBERT) as **drop-in replacements** for RoBERTa, and study the effect of **reducing/removing chunking** on long clinical notes.
- **Introduce three description-aware models**:
  - **PLM-DCA** (Description Cross-Attention) — initializes label queries from ICD code descriptions,
  - **PLM-DE** (Dual Encoding) — dynamically encodes descriptions alongside notes using the same encoder,
  - **PLM-RR** (Retrieval + Re-ranking) — retrieves top-K codes with PLM-CA, then performs **token-level cross-attention** between code descriptions tokens and clinical note tokens to re-rank candidates.

**Key findings.** Reducing/ removing chunking captures long-context dependencies across extended text particularly for hypertension related codes in the PLM-CA framework; integrating **ICD code descriptions** adds crucial semantics for **rare codes**. **PLM-RR** achieves **state-of-the-art** rare-code performance, improving **Macro-F1 (F1 with all classes weighted equally) by ~20%** and reducing **never-predicted-correctly** codes by **31.2%** vs. PLM-CA, while matching or surpassing it on other metrics.

**What’s in this repo.** Reproducible training/evaluation pipelines for PLM-CA, PLM-DCA, PLM-DE, and **PLM-RR**, plus configs and scripts to run with **ModernBERT/BioClinical-ModernBERT/NeoBERT** backbones and optional long-context (less/more chunking).


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
10. Prepare environment, dataset and models by running the command: `make prepare_everything`

You are now all set to run experiments!

# Note on licenses

## MIMIC
You need to obtain a non-commercial licence from physionet to use MIMIC. You will need to complete training. The training is free, but takes a couple of hours. - [link to data access](https://physionet.org/content/mimiciv/2.2/)

## Model weights can only be used non-commercially
While we would love to make everything fully open source, we cannot. Becaue MIMIC has a non-commercial license, the models trained using that data will also have a non-commercial licence. Therefore, using our models, RoBERTa-base-PM-M3-Voc and BioClinical ModernBERT weights for commercial usecases is forbidden.

# How to run experiments
## How to train a model
The following commands are how to train each model in the dissertation, all experiments performed on MIMIC-IV ICD-10 full, GPU 0:
   * Train PLM-CA with RoBERTa-pm-M3-Voc: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd model=plm_icd gpu=0 dataloader.max_batch_size=1 data=mimiciv_icd10`
   * Train PLM-CA with ModernBERT, NeoBERT or BioClinical ModernBERT on a chunk size of 128. Just replace model.path with desired model: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd model=plm_icd_modernbert model.configs.model_path=models/modernbert-base gpu=0 dataloader.max_batch_size=1 data=mimiciv_icd10 model.configs.chunk_size=128`
   * Train PLM-DCA on MIMIC-IV ICD-10 full on GPU 0: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd model=plm_icd gpu=0 dataloader.max_batch_size=1 data=mimiciv_icd10 model.configs.init_with_descriptions=true`
   * Train PLM-DE on MIMIC-IV ICD-10 full on GPU 0: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd model=plm_icd data=mimiciv_icd10 gpu=CUDA_VISIBLE_DEVICES dataloader.max_batch_size=16 model.configs.cross_attention=true trainer.epochs=20 callbacks.2.configs.patience=20 dataloader.num_workers=32 model.configs.attention_type="dual_encoding"`
   * Lastly to train PLM-RR, the frozen retriever must be loaded from a saved checkpoint that any of the above experiments will save to "/models/", replace PATH_TO_CHECKPOINTED_MODEL with this checkpoint: `poetry run python train_plm.py experiment=mdace_icd9_code/plm_icd model=plm_icd data=mimiciv_icd10 gpu=0 dataloader.max_batch_size=2 trainer.epochs=20 callbacks.2.configs.patience=6 dataloader.num_workers=32 model.configs.attention_type="token_level" top_k=300 load_model=/models/PATH_TO_CHECKPOINTED_MODEL`

# Overview of the repository (inherited from https://github.com/JoakimEdin/explainable-medical-coding)
#### configs
We use [Hydra](https://hydra.cc/docs/intro/) for configurations. The configs for every experiment is found in `explainable_medical_coding/configs/experiments`.

#### data
This is where the splits and datasets are stored

#### models
The directory contains the model weights.

#### notebooks
analysis.ipynb is the jupyter notebook used for the results analysis done in the dissertation.

#### explainable_medical_coding
This is were the code for running the experiments.

# My setup
All experiements were ran on a single A100 80GB per experiment.

# Acknowledgement
Thank you again to Joakim Edin and contributers for making their [Explainable Medical Coding](https://github.com/JoakimEdin/explainable-medical-coding) available. This made experiementation much easier.
