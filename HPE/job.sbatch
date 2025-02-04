#!/bin/bash
#SBATCH --job-name=job
#SBATCH --output=output.txt
#SBATCH --error=errors.txt
#SBATCH --ntasks=1								# quanti task eseguire
#SBATCH --nodes=1								# quanti nodi usare (ce ne sono 4 xcpu e 4 xgpu)
#SBATCH --partition=xgpu						# richiedi un nodo xgpu
#SBATCH --gres=gpu:tesla:1						# richiedi esplicitamente la tesla
#SBATCH --cpus-per-task=16						# quante cpu allocare ad ogni task (32 cpu cores per gpu node max)
#SBATCH --mem=44000								# quanta memoria da allocare

source /home/spoleto/.bashrc
/home/spoleto/.mambaforge/bin/activate hpe

python train_hpe.py --continue_training