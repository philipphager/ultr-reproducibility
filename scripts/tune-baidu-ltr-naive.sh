#!/bin/bash

#SBATCH --job-name=tune-baidu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=1-9

source ${HOME}/.bashrc
mamba activate ultr-reproducibility

HPARAMS_FILE=scripts/1_tune-dims.txt

srun python -u main.py -m \
  hydra.sweep.dir=/projects/0/prjs0860/hydra/tune \
  checkpoints=False \
  logging=True \
  data=baidu-mlm-ctr \
  model=naive-pointwise \
  model.config.features=ltr \
  max_epochs=15 \
    $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1)
