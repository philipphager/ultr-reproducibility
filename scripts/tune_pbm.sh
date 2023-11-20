#!/bin/bash -l

#SBATCH --job-name=tune-pbm
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64GB
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=p.k.hager@uva.nl

source ${HOME}/.bashrc
conda activate baidu-reproducibility

python main.py -m model=pbm loss=pointwise +tune=pbm
