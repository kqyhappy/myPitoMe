# DSAA5013 PiToMe Reproduction Project

## Introduction

This repository is for the DSAA5013 course project. It reproduces selected experiments from [Accelerating Transformers with Spectrum-Preserving Token Merging](https://arxiv.org/pdf/2405.16148), covering two tasks: **image-text retrieval** and **text classification**.

The paper proposes PiToMe, a token-merging method that accelerates Transformer models by merging redundant tokens while preserving informative token structure through a spectrum-preserving criterion. In this project, we compare PiToMe with ToMe as the baseline to evaluate the trade-off between computational cost and task performance.



## How to Run



Initialize the environment, including the conda environment, Python version, PyTorch backend, project dependencies from `requirements.txt`:

```bash
bash scripts/env/init_environment.sh
```


On a Slurm cluster, the same scripts can be submitted with `sbatch`:

```bash
sbatch scripts/env/init_environment.sh
```

## Text Classification

This part reproduces the text-classification task. The experiments evaluate PiToMe on BERT-Base with different token-retention ratios.


Then activate the environment and prepare datasets:

```bash
conda activate pitome
bash scripts/env/download_datasets.sh all
```

Run text-classification evaluation:

```bash
bash scripts/tasks/text_classification.sh pitome
bash scripts/tasks/text_classification.sh tome
```

On a Slurm cluster, the same scripts can be submitted with `sbatch`:

```bash
sbatch scripts/env/download_datasets.sh all
sbatch scripts/tasks/text_classification.sh pitome
sbatch scripts/tasks/text_classification.sh tome
```

### Datasets

- SST-2
- Rotten Tomatoes
- IMDb

### Model

- `bert-base-uncased`

### Main Results

Results are saved in `outputs/tc_outputs/`:

The figure compares the GFLOPs-accuracy trade-off of PiToMe and ToMe. The no-compression baseline is marked as a star. When repeated rows are present in the CSV logs, the latest row for each method and ratio is reported.

![FLOPs-accuracy tradeoff for PiToMe and ToMe on text classification](figures/tc_flops_accuracy_tradeoff.png)

The table reports the PiToMe FLOPs speedup, computed as `ratio 1.0 GFLOPs / current ratio GFLOPs`.

| Ratio | SST-2 FLOPs Speedup | SST-2 Acc. | Rotten FLOPs Speedup | Rotten Acc. | IMDb FLOPs Speedup | IMDb Acc. |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 1.00x | 92.52 | 1.00x | 96.78 | 1.00x | 92.65 |
| 0.9 | 1.29x | 91.85 | 1.30x | 96.42 | 1.32x | 92.26 |
| 0.8 | 1.76x | 90.18 | 1.77x | 94.56 | 1.84x | 91.68 |
| 0.7 | 2.43x | 86.38 | 2.44x | 91.34 | 2.66x | 90.37 |
| 0.6 | 3.51x | 81.70 | 3.55x | 83.44 | 3.97x | 87.11 |
| 0.5 | 5.22x | 73.66 | 5.38x | 72.90 | 6.23x | 81.69 |
