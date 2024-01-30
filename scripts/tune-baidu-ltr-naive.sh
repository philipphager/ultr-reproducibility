#!/bin/bash

#SBATCH --job-name=naive-ltr
#SBATCH --partition=genoa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28GB
#SBATCH --time=6:00:00
#SBATCH --array=1-18

source ${HOME}/.bashrc
mamba activate ultr-reproducibility

HPARAMS_FILE=scripts/3_tune-ltr.txt

srun python -u main.py -m \
  hydra.sweep.dir=/projects/0/prjs0860/hydra/tune \
  checkpoints=False \
  logging=True \
  data=ltr \
  model=naive-pointwise,naive-listwise \
  model.config.features=ltr \
  max_epochs=15 \
    $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1)
