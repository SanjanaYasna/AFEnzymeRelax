#!/bin/bash -l
#SBATCH --job-name=gconv
#SBATCH --output=gconv.txt
#SBATCH --partition bigjay
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=20G
#SBATCH --time=72:00:00

module load conda
conda activate graph
cd /kuhpc/work/slusky/syasna_sta/func_pred/code/GRPN_adam_gconv
export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' 
export 'PYTORCH_NO_CUDA_MEMORY_CACHING=1'
conda run -n graph python train.py