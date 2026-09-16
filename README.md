# Compressing the Attention Output Projection: an MLA inspired Extension

This repository contains a model pretraining and finetuning harness built specifically to test and compare a multi-headed latent attention (MLA) inspired compression/decompression mechanism applied to the attention output projection $W^{O}$, where attention heads are combined together.

## Table of Contents

1. [Introduction](#introduction)
2. [Attention Variants](#attention-variants)
3. [Repository layout](#repository-layout)
4. [Installation](#installation)
5. [Quickstart](#quickstart)
6. [Configurations](#configurations)
7. [Reproducing the experiments](#reproducing-the-experiments)
    1. [Pretraining](#pretraining)
    2. [Finetuning](#finetuning)
8. [Using DistributedDataParallel](#using-distributeddataparallel)
    1. [Pretraining](#pretraining-1)
    2. [Finetuning](#finetuning-1)
9. [Results](#results)
    1. [BERT](#bert)
        1. [Pretraining Results](#pretraining-results)
        2. [Finetuning Results](#finetuning-results)
    2. [GPT2](#gpt2)
        1. [Pretraining Results](#pretraining-results-1)
        2. [Finetuning Results](#finetuning-results-1)
10. [Findings / Discussion](#findings--discussion)
11. [Inference Time Analysis](#inference-time-analysis)
12. [Next Steps](#next-steps)
13. [References](#references)

## Introduction

After reading the DeepSeek-V2 paper I was impressed by the fact that their model managed to achieve a 93.3% reduction in the KV cache required per token, a huge saving compared to prior SOTA models using standard multi-head attention (MHA). I was even more impressed that DeepSeek-V2 outperformed its predecessor DeepSeek 67B and was competitive with SOTA models on evaluation benchmarks while having fewer activated parameters.

As such, this research explores whether the compress/decompress mechanism DeepSeek-V2 introduced on the reading side of attention (Q/KV latents) could also be applied to the writing side, the attention output projection $W^{O}$. Since a model produces several attention heads that may carry overlapping information, I ask if this redundancy could be removed via an additional compress/decompress mechanism applied to $W^{O}$. Thus, questioning if there is parameter redundancy in $W^{O}$ that can be removed, reducing the model's parameter count while maintaining performance? A potential knock-on effect is faster inference, by extending DeepSeek's inference time matrix absorption trick to $W^{O}$.

This repository was built to test exactly that. Initial experiments apply the compress/decompress mechanism to the BERT and GPT2 architectures, with a scaled down number of parameters. With the aim of identifying if it is possible to test whether my hypothesis holds with a small compute budget before potentially scaling to a larger model/data regime.

## Attention Variants

As part of this experimentation I wanted to compare four different attention mechanisms. These are described in the table below:

| Name | Description | Compresses |
|------|-------------|------------|
| Multi-head Attention (MHA) - baseline | Standard multi-head attention implementation. | - |
| Multi-head Attention Extension (MHAE) | Standard multi-head attention with a compress/decompress mechanism applied to the output projection $W^{O}$. | $W^{O}$ |
| Multi-head Latent Attention (MLA) - baseline | Basic multi-head latent attention implementation. | Q + KV |
| Multi-head Latent Attention Extension (MLAE) | Basic multi-head latent attention implementation with a compress/decompress mechanism applied to $W^{O}$. | Q + KV + $W^{O}$ |

The four variants also form a $2\times2$ matrix over read side (Q/KV) and write side ($W^O$) compression:

|              | dense $W^O$ | compressed $W^O$ |
|--------------|-------------|------------------|
| dense Q/KV   | MHA         | MHAE             |
| latent Q/KV  | MLA         | MLAE             |

Pretrained MHA and MLA variants serve as baselines against which the pretrained MHAE and MLAE attention variants are compared. To further test a model's capabilities on downstream tasks, pretrained models were additionally fine-tuned and evaluated on a number of the GLUE tasks. 

## Repository layout

```
├── src
│   └── mla
│       ├── config
│       │   ├── experiments
│       │   │   ├── BERT
│       │   │   │   ├── finetuning/       # per-task dirs: COLA, MNLI, MRPC, QNLI, QQP, RTE, SST2
│       │   │   │   └── pretraining/      # per-task dirs: COLA, MNLI, MRPC, QNLI, QQP, RTE, SST2  
│       │   │   └── GPT2
│       │   │       ├── finetuning/       # per-task dirs: COLA, MNLI, MRPC, QNLI, QQP, RTE, SST2
│       │   │       └── pretraining/      # per-task dirs: COLA, MNLI, MRPC, QNLI, QQP, RTE, SST2  
│       │   ├── __init__.py
│       │   ├── paths.py
│       │   └── schema.py
│       ├── model_training
│       │   ├── __init__.py
│       │   ├── model_finetuning.py
│       │   ├── model_pretraining.py
│       │   ├── model_training.py
│       │   ├── run_glue_benchmark.py
│       │   └── strategies.py
│       ├── models
│       │   ├── BERT                      # attention.py, encoder, layer, heads, ...
│       │   ├── GPT2                      # attention.py, block, mlp, heads, ...
│       │   └── __init__.py
│       └── utils
│           ├── __init__.py
│           ├── attention_hooks.py
│           ├── attention_utils.py
│           ├── data_preparation.py
│           ├── checkpointing.py
│           ├── metrics.py
│           ├── profiling.py
│           ├── progress.py
│           ├── reporting.py
│           └── setup.py
├── .gitignore
├── .python-version
├── docs
│   └── output_latent_absorption.md
├── notebooks                             # SVD analysis of W^O redundancy (DeepSeek-V2, GPT2 (multiple scales), BERT)
├── pyproject.toml
├── README.md
└── uv.lock  
```

For each model's pretraining directory, `experiments/{model}/pretraining/` there exists a YAML file per attention mechanism `{MHA, MHAE, MLA, MLAE}.` Similarly, for each model's finetuning directory `experiments/{model}/finetuning/`, there is an additional GLUE task directory `{COLA, MNLI, MRPC, QNLI, QQP, RTE, SST2}` with each of these directories containing a YAML file per attention mechanism, named `tiny{model}_{attention_architecture}_{task}.yaml`.

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) which auto-installs Python 3.13.14. Training is best run on a GPUs, while this harness also supports multi-GPU training runs via Distributed Data Parallel (DDP), for more details please see section  [Distributed Data Parallel](#using-distributeddataparallel). However, if no GPU is available, models can be trained on a CPU (albeit very slowly).

W\&B is used to log model parameters/gradients and experiment results. To use it, set the environent variable `WANDB_PW` equal to your W\&B API key, otherwise W\&B will run in offline mode.

To install this project please follow these steps:

```bash
# 1. Clone repository
git clone git@github.com:DavidJohnQuinlan/MLA_Output_Projection_Extension.git
cd MLA_Output_Projection_Extension

# 2. Create a local virtual environment
uv sync
```

## Quickstart

Once everything is installed you can start utilising this repository. Step one requires the pretraining of a model before finetuning in step two. One can finetune on a specific task or run the full suite of GLUE tasks. The commands to run quick, small budget pretraining and finetuning tasks are shown below for the TinyGPT2 MHA architecture. The more involved commands required to pretrain and finetune `MHAE`, `MLA` and `MLAE` are detailed in the [Pretraining](#pretraining) and [Finetuning](#finetuning) sections below.

```bash
# 1. (Required) Start pretraining GPT2 MHA on a small training budget
uv run pretraining --config-name=GPT2/pretraining/tinygpt2_mha \
  dataset_name=Salesforce/wikitext dataset_config_name=wikitext-2-raw-v1 \
  max_load_pct=null train_token_budget=10_000 val_token_budget=1_000 \
  max_steps=25 train_eval_steps=5 eval_steps=5 'pretraining_seeds=[42]'

# 2. Once completed pretraining start finetuning on a single GLUE task e.g. CoLA with a small training budget
uv run finetuning --config-name=GPT2/finetuning/COLA/tinygpt2_mha_cola \
  max_steps=25 train_eval_steps=5 eval_steps=5 \
  pretraining_seed=42 'finetuning_seeds=[42]'

# 3. Run finetuning on all GLUE tasks, with a small training budget
uv run run_glue_benchmark --base_model GPT2 --architecture MHA \
  max_steps=25 train_eval_steps=5 eval_steps=5 \
  pretraining_seed=42 'finetuning_seeds=[42]'
```

It is worth noting that there are three main entry points to pretrain/finetune models:
- `pretraining` → `model_pretraining:model_pretraining`
- `finetuning` → `model_finetuning:run_finetuning`
- `run_glue_benchmark` → `run_glue_benchmark:main`

## Configurations

Hydra is used to manage the pretraining and finetuning configurations. There is one pretraining config per attention architecture (`MHA`, `MHAE`, `MLA`, `MLAE`), while for finetuning each GLUE task has four configs, one for each attention architecture. Hydra allows any config value to be overridden via the command line, making it easy to sweep model hyperparameters. The main hyperparameters that vary across experiments are the compression dimensions, `kv_compression_dim`, `q_compression_dim` and `output_compression_dim`. These values `kv`, `q` and `o` represent the latent dimensions the model compresses down to. The smaller the value, the greater the compression, the more parameters removed from the model.

Beyond model hyperparameters, each config holds all model training, evaluation and dataset settings, so a run is fully specified by and reproducible from its configuration. Configs are validated against their corresponding schemas `PreTrainingConfig` / `FineTuningConfig`, which flag any missing or incorrectly typed fields.

## Reproducing the experiments

As mentioned above, the configuration is the source of truth in this repository, here the model is defined, either GPT2 or BERT, along with its hyperparameters `kv_compression_dim`, `q_compression_dim` and `output_compression_dim`. For a dataset, `openwebtext` was chosen given it is a large dataset which can be filtered to select enough tokens to perform one full training run i.e. only one epoch. However, the harness is built to handle other datasets e.g. `WikiText-103`, which was used to train the TinyBERT experiments.

The general flow of the tasks in this repository is as follows:

1. Prior to model pretraining or finetuning, the requested dataset, as defined in the configuration, will be downloaded from HuggingFace, prepared, sliced if necessary and tokenized before being saved locally in a model specific path with the dataset's configuration parameters encoded into the directory name where it is saved e.g. `TinyGPT2/pretraining/Skylion007__openwebtext_ac1e385807/`.
2. Before finetuning, one can pretrain a language model TinyBERT or GPT2 (small scale) on a previously defined and downloaded dataset. Once pretraining is completed the results are stored in `pretraining_results.csv` in the `/training_models/{model}/pretraining/` directory. Best model checkpoints and their associated configuration files are also stored in this location.
3. Once a model has been pretrained, it is possible to finetune a pretrained model on some downstream task. It is possible to finetune on an individual GLUE task or the full set of GLUE tasks. Similar to above, this will kick off data download and preparation before finetuning the model on this dataset. Finetuning is repeated for five seed values with the average results stored in the `finetuning_results.csv` file located in the `/training_models/{model}/finetuning/` directory. Each task's best model checkpoint and configuration are stored in a task specific subdirectory e.g. `training_models/TinyGPT2/finetuning/cola`.

### Pretraining

When wishing to pretrain a model, one should use the following command (a template, followed by a concrete example):

```bash
uv run pretraining --config-name={MODEL_NAME}/pretraining/{model_name}_{attention_architecture} \
  kv_compression_dim={kv_compression_dim} q_compression_dim={q_compression_dim} output_compression_dim={output_compression_dim}

uv run pretraining --config-name=GPT2/pretraining/tinygpt2_mlae \
  kv_compression_dim=64 q_compression_dim=64 output_compression_dim=64
```

With the parameters `kv_compression_dim`, `q_compression_dim` and `output_compression_dim` optional depending on the attention architecture.

### Finetuning

Similarly, when one wishes to finetune a previously pretrained model on a single GLUE task, they should use the following command (template and concrete example):

```bash
uv run finetuning --config-name={MODEL_NAME}/finetuning/{GLUE_TASK}/{model_name}_{attention_architecture}_{glue_task} \
  kv_compression_dim={kv_compression_dim} q_compression_dim={q_compression_dim} output_compression_dim={output_compression_dim} \
  pretraining_seed={pretraining_seed}

uv run finetuning --config-name=GPT2/finetuning/COLA/tinygpt2_mlae_cola \
  kv_compression_dim=64 q_compression_dim=64 output_compression_dim=64 \
  pretraining_seed=42
```

And in the case that one wishes to run the full suite of GLUE tasks, use the following command (template and concrete example):

```bash
uv run run_glue_benchmark --base_model {MODEL_NAME} --architecture {ATTENTION_ARCHITECTURE} \
  --kv {kv_compression_dim} --q {q_compression_dim} --o {output_compression_dim} \
  --pretraining_seed {pretraining_seed}

uv run run_glue_benchmark --base_model GPT2 --architecture MLAE \
  --kv 64 --q 64 --o 64 --pretraining_seed 42
```

With the parameters `kv_compression_dim`, `q_compression_dim` and `output_compression_dim` optional depending on the attention architecture.

## Using DistributedDataParallel

To support the scaling of experiments to larger models and datasets, this harness combines HuggingFace Accelerate with Distributed Data Parallel (DDP), enabling multi-GPU training with mixed precision and gradient accumulation. The commands below launch pretraining or finetuning runs across multiple GPU devices.

### Pretraining

To pretrain a model on multiple GPU devices, use the following command (template and example command):
```bash
uv run accelerate launch --multi_gpu --num_processes={num_processes} \
  -m mla.model_training.model_pretraining \
  --config-name={MODEL_NAME}/pretraining/{model_name}_{attention_architecture} \
  kv_compression_dim={kv_compression_dim} q_compression_dim={q_compression_dim} output_compression_dim={output_compression_dim} \
  'pretraining_seeds=[{pretraining_seed}]'

uv run accelerate launch --multi_gpu --num_processes=2 \
  -m mla.model_training.model_pretraining \
  --config-name=GPT2/pretraining/tinygpt2_mlae \
  kv_compression_dim=64 q_compression_dim=64 output_compression_dim=64 \
  'pretraining_seeds=[42]'
```
### Finetuning

To finetune a model on a single GLUE task across multiple GPUs, use the following command (template and example command):
```bash
uv run accelerate launch --multi_gpu --num_processes={num_processes} \
  -m mla.model_training.model_finetuning \
  --config-name={MODEL_NAME}/finetuning/{GLUE_TASK}/{model_name}_{attention_architecture}_{glue_task} \
  kv_compression_dim={kv_compression_dim} q_compression_dim={q_compression_dim} output_compression_dim={output_compression_dim} \
  pretraining_seed={pretraining_seed} 'finetuning_seeds=[{finetuning_seeds}]'

uv run accelerate launch --multi_gpu --num_processes=2 \
  -m mla.model_training.model_finetuning \
  --config-name=GPT2/finetuning/COLA/tinygpt2_mlae_cola \
  kv_compression_dim=64 q_compression_dim=64 output_compression_dim=64 \
  pretraining_seed=42 'finetuning_seeds=[42]'
```

To run the suite of GLUE finetuning tasks across multiple GPUs, use the following command (template and example):
```bash
uv run accelerate launch --multi_gpu --num_processes={num_processes} \
  -m mla.model_training.run_glue_benchmark \
  --base_model {MODEL_NAME} --architecture {ATTENTION_ARCHITECTURE} \
  --kv {kv_compression_dim} --q {q_compression_dim} --o {output_compression_dim} \
  --pretraining_seed {pretraining_seed} 'finetuning_seeds=[{finetuning_seeds}]'

uv run accelerate launch --multi_gpu --num_processes=2 \
  -m mla.model_training.run_glue_benchmark \
  --base_model GPT2 --architecture MLAE \
  --kv 64 --q 64 --o 64 \
  --pretraining_seed 42 'finetuning_seeds=[42]'
```

## Results

The majority of model training was conducted on a personal RTX 2080 Ti GPU for simplicity and cost reasons. Model training was performed at a small scale on both a BERT and GPT2 model. The details of each model's pretraining and finetuning can be found below. For more precise details of model hyperparameters and data configurations, please inspect the relevant pretraining or finetuning configuration files.

CKA stands for Centered Kernel Alignment, which measures representational similarity between attention heads. The higher the CKA score, the more similar a model layer's attention heads are, i.e. more redundant heads.

### BERT 
**Model:** 
  - TinyBERT
  - 4 layers
  - 12 heads
  - hidden size 312
  - ~14.38M parameters for the MHA baseline 

**Pretraining:** 
  - WikiText-103 (wikitext-103-raw-v1),
  - masked language modelling (15% masking)
  - sequence length 128
  - 25k steps 
  - learning rate 5e-4

**Finetuning:** 
- GLUE, 5 seeds per task; mean ± std reported. 
- Each task uses its standard metric (CoLA → Matthews corr, MRPC/QQP → F1, all others → accuracy).

#### Pretraining Results

| Variant                     | Params | GFLOPS  | Val Loss | Val PPL | MLM Acc | CKA    |
|-----------------------------|--------|---------|----------|---------|---------|--------|
| MHA  (baseline)             | 14.38M | 28.843  | 2.6618   | 14.3221 | 0.5213  | 0.1117 |
| MHAE (o=32)                 | 14.07M | 28.209  | 2.8784   | 17.7856 | 0.4933  | 0.1099 |
| MLA  (kv=32, q=32)          | 13.41M | 26.870  | 3.3187   | 27.6244 | 0.4344  | 0.2673 |
| MLAE (kv=32, q=32, o=32)    | 13.10M | 26.234  | 3.3100   | 27.3852 | 0.4359  | 0.2778 |
| MHAE (o=64)                 | 14.15M | 28.374  | 2.7697   | 15.9544 | 0.5095  | 0.1272 |
| MLA  (kv=64, q=64)          | 13.61M | 27.277  | 3.1098   | 22.4172 | 0.4620  | 0.3026 |
| MLAE (kv=64, q=64, o=64)    | 13.38M | 26.808  | 3.1079   | 22.3746 | 0.4613  | 0.2538 |
| MHAE (o=128)                | 14.31M | 28.702  | 2.6947   | 14.8006 | 0.5169  | 0.1257 |
| MLA  (kv=128, q=128)        | 14.01M | 28.098  | 2.8312   | 16.9659 | 0.4985  | 0.2333 |
| MLAE (kv=128, q=128, o=128) | 13.95M | 27.957  | 2.8395   | 17.1064 | 0.4955  | 0.2321 |


#### Finetuning Results

| Variant                     | CoLA (MCC)      | MRPC (F1)       | RTE (acc)       | SST-2 (acc)     | QNLI (acc)      | QQP (F1)        | MNLI (acc)      | Avg Score       |
|-----------------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| MHA  (baseline)             | 0.2159 ± 0.0135 | 0.8258 ± 0.0045 |	0.5726 ± 0.0252	| 0.8644 ± 0.0044 | 0.8046 ± 0.0031	| 0.8020 ± 0.0008	| 0.6997 ± 0.0020	| 0.6836 ± 0.0056 |
| MHAE (o=32)                 | 0.1964 ± 0.0184	| 0.8191 ± 0.0043	| 0.5451 ± 0.0174	| 0.8372 ± 0.0034	| 0.7992 ± 0.0022	| 0.7843 ± 0.0013	| 0.6739 ± 0.0031	| 0.6650 ± 0.0044 |
| MLA (k=32, q=32)            | 0.1485 ± 0.0162	| 0.8194 ± 0.0052	| 0.5502 ± 0.0106	| 0.8392 ± 0.0038	| 0.6821 ± 0.0017	| 0.7356 ± 0.0046	| 0.6215 ± 0.0023	| 0.6281 ± 0.0028 |
| MLAE (kv=32, q=32, o=32)    | 0.1270 ± 0.0210	| 0.8169 ± 0.0022	| 0.5473 ± 0.0128 |	0.8433 ± 0.0021	| 0.6773 ± 0.0081 |	0.7458 ± 0.0039	| 0.6462 ± 0.0005	| 0.6291 ± 0.0034 |
| MHAE (o=64)                 | 0.2041 ± 0.0105	| 0.8267 ± 0.0031	| 0.5365 ± 0.0239	| 0.8445 ± 0.0028 |	0.8005 ± 0.0017	| 0.7920 ± 0.0013	| 0.6777 ± 0.0044 |	0.6689 ± 0.0028 |
| MLA  (kv=64, q=64)          | 0.1368 ± 0.0195	| 0.8177 ± 0.0042	| 0.5321 ± 0.0132	| 0.8408 ± 0.0054	| 0.7210 ± 0.0150	| 0.7680 ± 0.0009	| 0.6704 ± 0.0014	| 0.6410 ± 0.0055 |
| MLAE (kv=64, q=64, o=64)    | 0.1604 ± 0.0189	| 0.8197 ± 0.0042	| 0.5285 ± 0.0098	| 0.8472 ± 0.0052	| 0.7808 ± 0.0024	| 0.7805 ± 0.0019	| 0.6764 ± 0.0012	| 0.6562 ± 0.0016 |
| MHAE (o=128)                | 0.2040 ± 0.0240	| 0.8179 ± 0.0075	| 0.5401 ± 0.0109	| 0.8502 ± 0.0016	| 0.7975 ± 0.0025	| 0.7952 ± 0.0008	| 0.6837 ± 0.0031	| 0.6698 ± 0.0033 |
| MLA  (kv=128, q=128)        | 0.1910 ± 0.0115	| 0.8253 ± 0.0024	| 0.5365 ± 0.0221	| 0.8516 ± 0.0039	| 0.8013 ± 0.0040	| 0.7969 ± 0.0010	| 0.6890 ± 0.0020	| 0.6702 ± 0.0034 |
| MLAE (kv=128, q=128, o=128) | 0.1872 ± 0.0138	| 0.8219 ± 0.0039	| 0.5343 ± 0.0110	| 0.8585 ± 0.0012	| 0.8119 ± 0.0023	| 0.7871 ± 0.0034	| 0.6865 ± 0.0012	| 0.6696 ± 0.0037 |


### GPT2
**Model:** 
  - TinyGPT2
  - 6 layers
  - 4 heads
  - hidden size 312
  - ~22.9M parameters for the MHA baseline

**Pretraining:** 
  - OpenWebText
  - 1.4B tokens training budget
  - sequence length 512
  - one epoch (20k steps)
  - seed 42
  - learning rate 5e-4

**Finetuning:** 
  - GLUE, 5 seeds per task; mean ± std reported. 
  - Each task uses its standard metric (CoLA → Matthews corr, MRPC/QQP → F1, all others → accuracy).

#### Pretraining Results

| Variant                     | Params | GFLOPS  | Val Loss | Val PPL | Val Next Token Acc | CKA    |
|-----------------------------|--------|---------|----------|---------|--------------------|--------|
| MHA  (baseline)             | 22.87M | 185.951 | 3.9139   | 50.0935 | 0.3318             | 0.3256 |
| MHAE (o=16)                 | 22.35M | 181.659 | 4.0954   | 60.0624 | 0.3145             | 0.4042 |
| MLA  (kv=16, q=16)          | 21.27M | 172.828 | 4.2646   | 71.1364 | 0.2864             | 0.5209 |
| MLAE (kv=16, q=16, o=16)    | 20.75M | 168.536 | 4.3060   | 74.1445 | 0.2821             | 0.5333 |
| MHAE (o=32)                 | 22.41M | 182.152 | 4.0028   | 54.7530 | 0.3240             | 0.4364 |
| MLA  (kv=32, q=32)          | 21.42M | 174.059 | 4.1033   | 60.5424 | 0.3079             | 0.4886 |
| MLAE (kv=32, q=32, o=32)    | 20.96M | 170.259 | 4.1092   | 60.8987 | 0.3078             | 0.5652 | 
| MHAE (o=64)                 | 22.53M | 183.137 | 3.9506   | 51.9685 | 0.3288             | 0.4636 |
| MLA  (kv=64, q=64)          | 21.72M | 176.520 | 3.9941   | 54.2762 | 0.3219             | 0.4867 |
| MLAE (kv=64, q=64, o=64)    | 21.38M | 173.706 | 4.0110   | 55.2046 | 0.3200             | 0.5287 |
| MHAE (o=128)                | 22.77M | 185.108 | 3.9292   | 50.8655 | 0.3301             | 0.4444 |
| MLA  (kv=128, q=128)        | 22.32M | 181.443 | 3.9494   | 51.9066 | 0.3275             | 0.4542 |
| MLAE (kv=128, q=128, o=128) | 22.22M | 180.600 | 3.9625   | 52.5884 | 0.3261             | 0.4790 |
| MLAE (kv=128, q=128, o=16)  | 21.80M | 177.151 | 4.0907   | 59.7791 | 0.3140             | 0.4644 |
| MLAE (kv=128, q=128, o=32)  | 21.86M | 177.644 | 4.0220   | 55.8102 | 0.3206             | 0.4845 |
| MLAE (kv=128, q=128, o=64)  | 21.98M | 178.629 | 3.9768   | 53.3470 | 0.3254             | 0.4855 |


#### Finetuning Results

| Variant                     | CoLA (MCC)      | MRPC (F1)       | RTE (acc)       | SST-2 (acc)     | QNLI (acc)      | QQP (F1)        | MNLI (acc)      | Avg Score       |
|-----------------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| MHA  (baseline)             | 0.2557 ± 0.0267	| 0.8510 ± 0.0050	| 0.6361 ± 0.0124	| 0.8807 ± 0.0040	| 0.8135 ± 0.0048	| 0.8143 ± 0.0007	| 0.7304 ± 0.0012	| 0.7117 ± 0.0034 |
| MHAE (o=16)                 | 0.2243 ± 0.0098	| 0.8213 ± 0.0065	| 0.5697 ± 0.0167	| 0.8686 ± 0.0041	| 0.7844 ± 0.0059	| 0.7927 ± 0.0011	| 0.7054 ± 0.0021	| 0.6809 ± 0.0023 |
| MLA  (kv=16, q=16)          | 0.1694 ± 0.0077	| 0.8199 ± 0.0038	| 0.5675 ± 0.0117	| 0.8606 ± 0.0031	| 0.6584 ± 0.0044	| 0.7261 ± 0.0039	| 0.6510 ± 0.0009	| 0.6361 ± 0.0028 |
| MLAE (kv=16, q=16, o=16)    | 0.1691 ± 0.0179	| 0.8203 ± 0.0041	| 0.5733 ± 0.0260	| 0.8764 ± 0.0038	| 0.6542 ± 0.0049	| 0.7257 ± 0.0020	| 0.6485 ± 0.0020	| 0.6382 ± 0.0056 |
| MHAE (o=32)                 | 0.2253 ± 0.0162	| 0.8343 ± 0.0135	| 0.6043 ± 0.0106	| 0.8743 ± 0.0046	| 0.8039 ± 0.0053	| 0.8023 ± 0.0013	| 0.7158 ± 0.0019	| 0.6943 ± 0.0034 |
| MLA  (kv=32, q=32)          | 0.1824 ± 0.0247	| 0.8187 ± 0.0065	| 0.5791 ± 0.0162	| 0.8798 ± 0.0038	| 0.6662 ± 0.0062	| 0.7562 ± 0.0029	| 0.6632 ± 0.0019	| 0.6494 ± 0.0038 |
| MLAE (kv=32, q=32, o=32)    | 0.1975 ± 0.0147	| 0.8142 ± 0.0021	| 0.5798 ± 0.0126	| 0.8693 ± 0.0085	| 0.7338 ± 0.0096	| 0.7677 ± 0.0030	| 0.6655 ± 0.0025	| 0.6611 ± 0.0031 |
| MHAE (o=64)                 | 0.2303 ± 0.0252	| 0.8499 ± 0.0039	| 0.6181 ± 0.0113	| 0.8835 ± 0.0028	| 0.8186 ± 0.0029	| 0.8087 ± 0.0007	| 0.7267 ± 0.0017	| 0.7051 ± 0.0041 |
| MLA  (kv=64, q=64)          | 0.2088 ± 0.0161	| 0.8384 ± 0.0028	| 0.6022 ± 0.0257	| 0.8764 ± 0.0034	| 0.7974 ± 0.0091	| 0.8005 ± 0.0010	| 0.7194 ± 0.0015	| 0.6919 ± 0.0048 |
| MLAE (kv=64, q=64, o=64)    | 0.2345 ± 0.0113	| 0.8231 ± 0.0026	| 0.5697 ± 0.0122	| 0.8748 ± 0.0034	| 0.7959 ± 0.0027	| 0.7935 ± 0.0017	| 0.7119 ± 0.0025	| 0.6862 ± 0.0028 |
| MHAE (o=128)                | 0.2484 ± 0.0119	| 0.8417 ± 0.0022	| 0.6209 ± 0.0114	| 0.8803 ± 0.0027	| 0.8179 ± 0.0037	| 0.8070 ± 0.0017	| 0.7233 ± 0.0011	| 0.7057 ± 0.0034 |
| MLA  (kv=128, q=128)        | 0.2319 ± 0.0207	| 0.8594 ± 0.0120	| 0.6354 ± 0.0139	| 0.8803 ± 0.0038	| 0.8144 ± 0.0044	| 0.8083 ± 0.0015	| 0.7276 ± 0.0014	| 0.7082 ± 0.0033 |
| MLAE (kv=128, q=128, o=128) | 0.2223 ± 0.0168	| 0.8170 ± 0.0026	| 0.5682 ± 0.0142	| 0.8837 ± 0.0047	| 0.7461 ± 0.0089	| 0.7828 ± 0.0021	| 0.6995 ± 0.0006	| 0.6742 ± 0.0032 |
| MLAE (kv=128, q=128, o=16)  | 0.2082 ± 0.0201	| 0.8153 ± 0.0033	| 0.5805 ± 0.0194	| 0.8679 ± 0.0055	| 0.7936 ± 0.0034	| 0.7900 ± 0.0009	| 0.7067 ± 0.0019	| 0.6803 ± 0.0048 |
| MLAE (kv=128, q=128, o=32)  | 0.2096 ± 0.0157	| 0.8166 ± 0.0043	| 0.5819 ± 0.0084	| 0.8849 ± 0.0045	| 0.7018 ± 0.0120	| 0.7905 ± 0.0014	| 0.7036 ± 0.0030	| 0.6698 ± 0.0011 |
| MLAE (kv=128, q=128, o=64)  | 0.2280 ± 0.0145	| 0.8532 ± 0.0081	| 0.6195 ± 0.0071	| 0.8794 ± 0.0050	| 0.8152 ± 0.0045	| 0.8022 ± 0.0021	| 0.7205 ± 0.0018	| 0.7026 ± 0.0034 |

## Findings / Discussion
1. From the initial research notebooks, it is clear when examining DeepSeek-V2, GPT2 (at different scales) and BERT, that there exists parameter redundancy in the model and, of particular relevance to this investigation, in the output attention weights $W^{O}$. This was confirmed via the SVD analysis on the raw model's $W^{O}$ weights as well as the $W^{O}$ activations after passing some data through each model. Therefore, this made it a worthwhile investigation to undertake.
2. Initial TinyBERT experiment results appeared promising, with compressed attention variants approaching MHA baseline performance on GLUE tasks. As one can see from the above results, none of the experiments' average GLUE scores exceeded that of MHA. However, some variants did approach the MHA average GLUE score. At this model scale that felt promising, and so it seemed worth continuing the investigation, applying it to a pure decoder model, GPT2 (albeit also at a small scale due to time/cost constraints).
3. The TinyGPT2 experiments showed a clearer, more systematic trend. From the above results, one can see that no compression variant outperforms the MHA baseline. In general, as the latent compression dimensions decrease (compression increases), the pretraining validation perplexity rises, while the average GLUE score decreases. There are a few exceptions, but for TinyGPT2 at this scale, this is the general finding.
4. When exploring the results, it is interesting to see that none of the MLA architectures outperform MHA at this scale, although the average GLUE score does approach the MHA score as we reduce the level of compression on $kv$ and $q$ (higher $d_{kv}$ and $d_q$ values). This is contrary to the results published in DeepSeek-V2, however, those results were for a highly parameterised model.
5. There are a few instances of MHAE, MLA and MLAE at higher $d_{kv}$, $d_q$ and/or $d_o$ dimensions where the pretraining perplexity and average GLUE score approach the MHA baseline. In particular, for $d_{kv}=d_q=128$ (the best of the non-MHA variants), MLA achieves an average GLUE score of 0.7082 compared to the MHA baseline GLUE score of 0.7117. The same MLA model obtains a perplexity of 51.9, compared to MHA's 50.1.
6. MHAE models, which apply the compress–decompress mechanism only to the attention outputs, appear to outperform their corresponding MLA/MLAE models at higher levels of compression (low $d_o$). Comparing average GLUE scores, for $d_{kv}=d_q=d_o \in \{16, 32, 64\}$ MHAE outperforms both MLA and MLAE, while at $d_{kv}=d_q=d_o=128$ the results are very close. It would be interesting to explore why this is the case, why it appears not to hold once KV and output latents are combined, and whether it persists for larger, more parameterised models.
7. For the set of experiments that varied $d_o$ relative to $d_{kv}$ and $d_q$, it was interesting to note that setting all three values equal to each other did not return the optimal result. It was found that $d_{kv}=128, d_q=128, d_o=64$ outperformed its peers with $o \in \{16, 32, 128\}$. This was only a single result, however, it might be worth exploring further to see whether there is a relationship between $d_{kv}$, $d_q$ and the output latent dimension $d_o$.
8. From the GPT2 results, CKA tends to increase with compression, suggesting more overlap across attention heads (greater redundancy). This is consistent with the observed drop in performance, as heads become more similar, effective attention capacity falls. Notably, the effect is stronger for the MLA family than for MHAE at comparable dimensions (e.g. at dim 64, MHAE 0.46 vs MLA 0.49 vs MLAE 0.53, against an MHA baseline of 0.33), suggesting that compressing the read-side KV/Q latents collapses head diversity more than compressing the output projection alone.
9. Given that the TinyGPT2 model has approximately 22.9M parameters with 15.7M (~69%) embedding parameters dominating the overall parameter count, it is difficult to draw a true conclusion. The results at this model scale appear inconclusive, however, I would argue that the results are promising enough to merit further experimentation at a greater data/model scale. As shown above, the parameter savings were between 2.84% and 9.27%, these are modest because the embedding parameters dominate the total. A larger scale would give the model more attention parameters, attention heads and layers.

## Inference Time Analysis

The DeepSeek-V2 MLA paper detailed a matrix absorption trick, which enabled even greater inference time savings by folding the value up projection into the output projection, ensuring that the key and value vectors are never materialised. In this project, this was built upon further by extending the matrix absorption to the write side bottleneck $W^{O}$.

Theoretical results suggest that MLAE's absorbed operator is roughly $1.51-3.32\times$ smaller for the TinyGPT2 architecture, with potential for a greater reduction for larger scale models. The extension of the matrix absorption trick is independent of the KV cache reduction. However, it was shown that for wide models with many heads there is a saving in the total decode FLOPs.

Initial wall-clock latency measurements have been implemented, but they were applied to the training architecture rather than the optimised, matrix absorbed inference path, this is also the case for MLA, MHAE and MLAE. Benchmarking the absorbed-inference implementation is left as future work.

More details of this matrix absorption can be found in [`docs/output_latent_absorption.md`](docs/output_latent_absorption.md).

## Next Steps

1. Implement the absorption trick for the GPT2 models.
2. Scale the model to at least GPT2 small size and train it on an appropriate amount of data.
3. Implement a more modern decoder model.
4. If cost allows, leverage multiple GPU devices for model training to save time with larger-scale training.

## References

**Prior work / methods**
- DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* 2024. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) - the MLA read-side compress/decompress mechanism and inference-time matrix-absorption trick this project extends.
- Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) - encoder architecture (`bert-base-uncased` config, scaled down).
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. *Language Models are Unsupervised Multitask Learners.* 2019. [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - decoder architecture (`gpt2` config, scaled down).
- Kornblith, S., Norouzi, M., Lee, H., Hinton, G. *Similarity of Neural Network Representations Revisited (CKA).* ICML 2019. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414) - the metric used to measure attention-head redundancy.

**Datasets & benchmarks**
- Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., Bowman, S. *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding.* ICLR 2019. [arXiv:1804.07461](https://arxiv.org/abs/1804.07461) ([`nyu-mll/glue`](https://huggingface.co/datasets/nyu-mll/glue)).
- Merity, S., Xiong, C., Bradbury, J., Socher, R. *Pointer Sentinel Mixture Models (WikiText-103).* 2016. [arXiv:1609.07843](https://arxiv.org/abs/1609.07843).
- Gokaslan, A., Cohen, V. *OpenWebText Corpus.* 2019. [`Skylion007/openwebtext`](https://huggingface.co/datasets/Skylion007/openwebtext).

**Tooling**
- Wolf, T. et al. *Transformers: State-of-the-Art Natural Language Processing.* EMNLP 2020. [HuggingFace Transformers](https://github.com/huggingface/transformers).
- HuggingFace [Accelerate](https://github.com/huggingface/accelerate) — distributed / mixed-precision training.
- Yadan, O. *Hydra — A framework for elegantly configuring compflex applications.* 2019. [GitHub](https://github.com/facebookresearch/hydra).
- [calflops](https://github.com/MrYxJ/calculate-flops.pytorch) — FLOPs / parameter accounting.

See [`docs/output_latent_absorption.md`](docs/output_latent_absorption.md) for the full matrix-absorption derivation.
