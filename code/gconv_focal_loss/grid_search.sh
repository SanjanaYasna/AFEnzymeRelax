#!/bin/bash -l
#SBATCH --job-name=gconv_rest
#SBATCH --output=rest.txt
#SBATCH --partition bigjay
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=10G
#SBATCH --time=12:00:00

module load conda
conda activate graph
cd /kuhpc/work/slusky/syasna_sta/func_pred/code/gconv_focal_loss
export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' 
export 'PYTORCH_NO_CUDA_MEMORY_CACHING=1'
conda run -n graph python red_rest.py