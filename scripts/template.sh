#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=32:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu
#SBATCH --job-name={name}
#SBATCH --output=logs/{name}.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sarehal1@jhu.edu

module purge

singularity exec --nv \
	    --overlay /scratch/sa6875/singularity/grusha-babylm.ext3:ro \
	    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif\
	    /bin/bash -c "source /ext3/env.sh; python -u get_synsurp.py --model models/augment_0.5_{model}_sgd  --input ./data/items_{evalset}.pivot.csv --out surps/items_{evalset}.ambig.csv.m{model} --batch_size 1024 --model_type marginal_ambig --aligned --uncased --cuda"
