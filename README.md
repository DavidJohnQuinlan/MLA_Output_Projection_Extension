# Multi-head Latent Attention Extension

This repository is a model pretraining and finetuning harness built specifically to test and compare a multi-head latent attention (MLA) inspired compression/decompression mechanism applied to the attention output projection $W^{O}$, where attention heads are combined together.

## Introduction

After reading the DeepSeek V2 paper I was impressed by the fact that their model managed to achieve a 93.3% reduction in the KV cache required per token, a huge saving compared to prior SOTA models using standard multi-head attention (MHA). I was even more impressed that DeepSeek V2 outperformed its predecessor DeepSeek 67B and was competitive with SOTA models on evaluation benchmarks while having fewer activated parameters.

As such, this research explored my curiosity as to whether the compress/decompress mechanism DeepSeek V2 introduced could also be applied to the writing component of the model, the attention output projection $W^{O}$. Given a model produces a set of different attention heads, it is possible that these heads carry similar information, thus, we question if is it possible to remove any unnecessary redundant information via an additional compress/decompress mechanism applied to the attention output projection $W^{O}$.
This experiment thus tests whether $W^{O}$ compresson performance improvements could be made while reducing the number of parameters in the model. More precisely, is there parameter redundancy in the attention output projection matrix $W^{O}$ that could be removed? There is also the potential knock-on effect that, once implemented, this could improve inference time of the model too.

Therefore, this repository was built to test/compare if the inclusion of a compress/decompress mechanism when applied to the attention output projection matrix $W^{O}$ would improve a models performance while reducing its overall number of parameters. Initial experimentation involved applying this compress/decompress mechanism to the TinyBERT and (a scaled down) GPT2 architectures to identify if it was possible to prove with a small compute budget before potentially moving to a larger model/data scale.

## Attention Variants

As part of this experimentation I wanted to compare four different attention mechanisms. These are described in the table below:

| Name | Description | Compresses |
|------|-------------|------------|
| Multi-head Attention (MHA) | Standard multi-head attention implementation. | - |
| Multi-head Attention Extension (MHAE) | Standard multi-head attention with a compress/decompress mechanism applied to the output projection $W^{O}$. | $W^{O}$ |
| Multi-head Latent Attention (MLA) | Basic multi-head latent attention implementation. | Q + KV |
| Multi-head Latent Attention Extension (MLAE) | Basic multi-head latent attention implementation with a compress/decompress mechanism applied to $W^{O}$. | Q + KV + $W^{O}$ |

Experiments were run to create baseline benchmarks for pretrained MHA and MLA variants for which to compare pretrained MHAE and MLAE attention variations against. To further test a models capabilities on downstream tasks, pretrained models will be additionally fine-tuned and evaluated on a number of the GLUE tasks. 

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
│           ├── model_utils.py
│           └── utils.py
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock  
```

For each model's pretraining directory, `experiments/{model}/pretraining/` there exists a YAML file per attention mechanism `{MHA, MHAE, MLA, MLAE}.` Similarly, for each models finetuning directory `experiments/{model}/finetuning/`, there is an additional GLUE task directory `{COLA, MNLI, MRPC, QNLI, QQP, RTE, SST2}` with each of these directories containing a YAML file per attention mechanism, named `tiny{model}_{attention_architecture}_{task}.yaml`.

## Installation & Quickstart

This project uses [`uv`](https://docs.astral.sh/uv/) and requires Python 3.13.14 or greater. 

To install this project please follow these steps:

```bash
# 1. Clone repository
git clone git@github.com:DavidJohnQuinlan/Multihead_Latent_Attention_Extension.git
cd Multihead_Latent_Attention_Extension

# 2. Create a local virtual environment
uv sync
```

## Quickstart

Once everything is installed you can start utilising this repository. Step one requires the pretraining of a model before any finetuning can happen in step two. One can finetune on a specific task or run the full suite of GLUE tasks. The commands to run the default options pretraining and finetuning are shown below for the TinyGPT2 MHA architecture.

```bash
# 1. (Required) Start pretraining GPT2 MHA
uv run pretrain --config-name GPT2/pretraining/tinygpt2_mha

# 2. Once completed pretraining start finetuning on a single GLUE task e.g. CoLA
uv run finetune --config-name GPT2/finetuning/COLA/tinygpt2_mha_cola

# 3. Run finetuning on all GLUE tasks
uv run run_glue_benchmark --base_model GPT2 --architecture MHA
```

It is worth noting that there are three main entry points to pretrain/finetune models:
- `pretraining` → `model_pretraining:model_pretraining`
- `finetuning` → `model_finetuning:run_finetuning`
- `run_glue_benchmark` → `run_glue_benchmark:main`

## Configurations

For configuration management I decided to use Hydra, this allows for easy management of pretraining and finetuning configs. A config was created for each of the four attention architectures, `MHA`, `MHAE`, `MLA` and `MLAE` for pretraining. While for finetuning, each GLUE task has four configs one for each of the above mentioned attention architectures. Hydra allows the passing of arguments, via the command line, to a configuration file enabling easy experimentation with different sets of model hyperparameters. The main hyperparameters which we wish to vary for experimentation purposes are the key/value latent dimension `kv_compression_dim`, the query latent dimension `q_compression_dim` and the output projection latent dimension `output_compression_dim`. These values define the size of the latent space we will compress the respective attention `kv`, `q` or `output` values down to, with the smaller the value the greater the compression and the greater number of parameters removed from the model.

In addition to model hyperparameters, the configuration contains all model training and evaluation parameters and all dataset configurations. This repository was built with reproducibility in mind so users can change any config value as per their experimentation requirements. 

## Reproducing the experiments

As mentioned above the configuration is the place of truth in this repository, here we define the model GPT2 or BERT, its hyperparameters `kv_compression_dim`, `q_compression_dim` and `output_compression_dim`. For a dataset, I chose `openwebtext` given it is a large dataset which I can filter to select enough tokens to perform one full training run i.e. only one epoch.

The general flow of the tasks in this repository is as follows:

1. Prior to model pretraining or finetuning the requested dataset, as defined in the configuration, will be downloaded from HuggingFace, prepared, sliced if necessary and tokenized before being saved locally in a model specific path with the datasets configuration parameters encoded into the directory name where it is saved e.g. `TinyGPT2/pretraining/Skylion007__openwebtext_ac1e385807/`.
2. Before any finetuning, one can pretrain a language model TinyBERT or GPT2 (small scale) on a previously defined and downloaded dataset. Once pretraining is completed the results are stored in `pretrain_results.csv` in the `/training_models/TinyGPT2/pretraining/` directory. Best model checkpoints and their associated configuration files are also stored in this location.
3. Now that a model has been pretrained, it is possible to finetune a pretrained model on some downstream task. It is possible to finetune on either an individual GLUE task or the full set of GLUE tasks. Similarly to above, this will kick off data download and preparation before finetuning the model on this dataset. Finetuning is repeated for five seed values with the average results stored in the `finetune_results.csv` file located in the `/training_models/TinyGPT2/finetuning/` directory. Each task runs best model checkpoint and model configuration are stored in a task specific subdirectory e.g. `training_models/TinyGPT2/finetuning/cola`.

### Pretraining

The above commands ran the default pretraining and finetuning commands, however, when one wishes to pretrain a model, they should use the following command:

```bash
uv run pretraining --config-name={MODEL_NAME}/pretraining/{model_name}_{attention_architecture} \
  kv_compression_dim={kv_compression_dim} q_compression_dim={q_compression_dim} output_compression_dim={output_compression_dim}
```

With the parameters `kv_compression_dim`, `q_compression_dim` and `output_compression_dim` optional depending on the attention architecture.

### Finetuning

Similarly, when one wishes to finetune a previously pretrained model on a specific GLUE task, they should use the following command:

```bash
uv run finetuning --config-name={MODEL_NAME}/finetuning/{GLUE_TASK}/{model_name}_{attention_architecture}_{glue_task} \
  kv_compression_dim={kv_compression_dim} q_compression_dim={q_compression_dim} output_compression_dim={output_compression_dim}
```

And in the case that one wishes to run the full selection of GLUE tasks the following command structure needs to be followed:

```bash
uv run run_glue_benchmark --base_model {MODEL_NAME} --architecture {ATTENTION_ARCHITECTURE} \
  --pretraining_seed {pretraining_seed} --kv {kv_compression_dim} --q {q_compression_dim} --o {output_compression_dim}
```

With the parameters `kv_compression_dim`, `q_compression_dim` and `output_compression_dim` optional depending on the attention architecture.

## Using DistributedDataParallel

### Pretraining
```bash
uv run accelerate launch --multi_gpu --num_processes=2 \
  -m mla.model_training.model_pretraining \
  --config-name=GPT2/pretraining/tinygpt2_mha \
  dataset_name=Salesforce/wikitext dataset_config_name=wikitext-2-raw-v1 \
  max_load_pct=null train_token_budget=100_000 val_token_budget=10_000 \
  max_steps=100 train_eval_steps=10 eval_steps=10 \
  'pretraining_seeds=[42]'
```
### Finetuning

```bash
uv run accelerate launch --multi_gpu --num_processes=2 \
  -m mla.model_training.model_finetuning \
  --config-name=GPT2/finetuning/COLA/tinygpt2_mha_cola \
  pretraining_seed=42 max_steps=2 train_eval_steps=1 eval_steps=1 'finetuning_seeds=[42]'
```

### Testing

After making some changes one may wish to quickly test the harness to ensure that everything functions as expected. However, we do not want to pretrain on the full dataset, therefore, to simulate a pretraining run we can use the following command:

#### Pretraining

```bash
uv run pretraining --config-name={MODEL_NAME}/pretraining/{model_name}_{attention_architecture} \
  dataset_name=Salesforce/wikitext dataset_config_name=wikitext-2-raw-v1 \
  max_load_pct=null train_token_budget=100_000 val_token_budget=10_000 \
  max_steps=2 train_eval_steps=1 eval_steps=1 'pretraining_seeds=[42]'
```

#### Finetuning

After running the above pretraining command, we can simulate a single finetuning task by running the following command. It is necessary to tie the finetuning job to the pretraining seed.

```bash
uv run finetuning --config-name={MODEL_NAME}/finetuning/{GLUE_TASK}/{model_name}_{attention_architecture}_{glue_task} \
  pretraining_seed=42 max_steps=2 train_eval_steps=1 eval_steps=1 'finetuning_seeds=[42]'
```

## Results
### GPT2
**Model:** TinyGPT2 - 6 layers, 4 heads, hidden size 312 (~22.9M parameters for the MHA baseline). <br>
**Pretraining:** OpenWebText, 1.4B tokens training budget, sequence length 512, one epoch (20k steps), seed 42, learning rate 5e-4.  <br>
**Finetuning:** GLUE, 5 seeds per task; mean ± std reported. Each task uses its standard metric (CoLA → Matthews corr, MRPC/QQP → F1, all others → accuracy). <br>

#### Pretraining

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


#### Finetuning

| Variant                     | CoLA (MCC)      | MRPC (F1)       | RTE (acc)       | SST-2 (acc)     | QNLI (acc)      | QQP (F1)        | MNLI (acc)      | Avg Score       |
|-----------------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| MHA  (baseline)             | 0.2574 ± 0.0263 | 0.8425 ± 0.0069 | 0.6173 ± 0.0225 | 0.8764 ± 0.0048 | 0.8090 ± 0.0020 | 0.8138 ± 0.0008 | 0.7292 ± 0.0020 | 0.7117 ± 0.0038 |
| MHAE (o=16)                 | 0.2232 ± 0.0112 | 0.7844 ± 0.0147 | 0.5227 ± 0.0168 | 0.8642 ± 0.0039 | 0.7800 ± 0.0045 | 0.7915 ± 0.0011 | 0.7047 ± 0.0024 | 0.6809 ± 0.0026 |
| MLA  (kv=16, q=16)          | 0.1640 ± 0.0179 | 0.7543 ± 0.0160 | 0.5220 ± 0.0158 | 0.8560 ± 0.0050 | 0.6569 ± 0.0049 | 0.7211 ± 0.0032 | 0.6504 ± 0.0008 | 0.6361 ± 0.0032 |
| MLAE (kv=16, q=16, o=16)    | 0.1647 ± 0.0129 | 0.7731 ± 0.0044 | 0.5097 ± 0.0095 | 0.8718 ± 0.0030 | 0.6539 ± 0.0046 | 0.7188 ± 0.0016 | 0.6467 ± 0.0022 | 0.6382 ± 0.0063 |
| MHAE (o=32)                 | 0.2175 ± 0.0169 | 0.8134 ± 0.0258 | 0.5863 ± 0.0185 | 0.8677 ± 0.0055 | 0.7993 ± 0.0030 | 0.8017 ± 0.0015 | 0.7145 ± 0.0022 | 0.6943 ± 0.0038 |
| MLA  (kv=32, q=32)          | 0.1789 ± 0.0330 | 0.7669 ± 0.0246 | 0.5617 ± 0.0197 | 0.8761 ± 0.0057 | 0.6654 ± 0.0062 | 0.7531 ± 0.0035 | 0.6614 ± 0.0013 | 0.6494 ± 0.0042 |
| MLAE (kv=32, q=32, o=32)    | 0.1898 ± 0.0149 | 0.7621 ± 0.0111 | 0.5444 ± 0.0205 | 0.8638 ± 0.0065 | 0.7328 ± 0.0108 | 0.7655 ± 0.0032 | 0.6640 ± 0.0025 | 0.6611 ± 0.0035 | 
| MHAE (o=64)                 | 0.2288 ± 0.0245 | 0.8476 ± 0.0021 | 0.5870 ± 0.0189 | 0.8805 ± 0.0034 | 0.8135 ± 0.0016 | 0.8084 ± 0.0007 | 0.7254 ± 0.0020 | 0.7051 ± 0.0046 |
| MLA  (kv=64, q=64)          | 0.2076 ± 0.0107 | 0.8277 ± 0.0071 | 0.5877 ± 0.0186 | 0.8695 ± 0.0022 | 0.7930 ± 0.0068 | 0.7999 ± 0.0012 | 0.7181 ± 0.0013 | 0.6919 ± 0.0054 | 
| MLAE (kv=64, q=64, o=64)    | 0.2315 ± 0.0092 | 0.7665 ± 0.0235 | 0.5292 ± 0.0140 | 0.8700 ± 0.0038 | 0.7942 ± 0.0028 | 0.7929 ± 0.0017 | 0.7108 ± 0.0024 | 0.6862 ± 0.0032 | 
| MHAE (o=128)                | 0.2468 ± 0.0139 | 0.8326 ± 0.0035 | 0.6079 ± 0.0074 | 0.8771 ± 0.0047 | 0.8147 ± 0.0046 | 0.8061 ± 0.0015 | 0.7225 ± 0.0016 | 0.7057 ± 0.0038 |
| MLA  (kv=128, q=128)        | 0.2269 ± 0.0174 | 0.8519 ± 0.0169 | 0.6217 ± 0.0187 | 0.8752 ± 0.0066 | 0.8078 ± 0.0030 | 0.8078 ± 0.0017 | 0.7268 ± 0.0015 | 0.7082 ± 0.0037 |
| MLAE (kv=128, q=128, o=128) | 0.2164 ± 0.0192 | 0.7803 ± 0.0049 | 0.5271 ± 0.0100 | 0.8798 ± 0.0074 | 0.7452 ± 0.0087 | 0.7820 ± 0.0023 | 0.6986 ± 0.0009 | 0.6742 ± 0.0036 |
| MLAE (kv=128, q=128, o=16)  | 0.2078 ± 0.0269 | 0.7813 ± 0.0092 | 0.5545 ± 0.0220 | 0.8647 ± 0.0067 | 0.7896 ± 0.0019 | 0.7894 ± 0.0012 | 0.7059 ± 0.0014 | 0.6803 ± 0.0054 |
| MLAE (kv=128, q=128, o=32)  | 0.2055 ± 0.0126 | 0.7683 ± 0.0158 | 0.5466 ± 0.0106 | 0.8817 ± 0.0064 | 0.7014 ± 0.0125 | 0.7895 ± 0.0017 | 0.7025 ± 0.0029 | 0.6698 ± 0.0012 |
| MLAE (kv=128, q=128, o=64)  | 0.2159 ± 0.0157 | 0.8468 ± 0.0101 | 0.6072 ± 0.0110 | 0.8743 ± 0.0049 | 0.8122 ± 0.0076 | 0.8018 ± 0.0020 | 0.7193 ± 0.0009 | 0.7026 ± 0.0038 |


## TODO: Benchmarks / analysis tools
