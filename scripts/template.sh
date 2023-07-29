#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=32:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu
#SBATCH --job-name={name}
#SBATCH --output=logs/{name}.log

python -u get_synsurp.py --model models/augment_0.5_{model}_sgd  --input ./data/items_{evalset}.pivot.csv --out surps/items_{evalset}.ambig.csv.m{model} --batch_size 1024 --model_type marginal_ambig --aligned --uncased --cuda"
