# CDM Project Group 2

This project looks into calculating stylistic and lexical alignment in multi-party multi-gender dialogues using generalised linear models.

## Gettin Started

Make sure to activate the conda environment `cdm` (or first create one with `environment.yml`).

## Calculating Alignment
To calculate stylistic alignment, run `stylistic.py` with the arguments for dataset and the type of experiment to perform. For example,
```
python stylistic.py AMI 1
```
calculates performs Experiment 1 for stylistic alignment for the AMI dataset.

It is also possible to perform Experiment 1 for stylistic alignment per linguistic category. To do so, set the `--cat` argument to `True`. 

To calculate lexical alignment, run ` lexica.py` with the arguments for dataset and type of experiment, as before.

