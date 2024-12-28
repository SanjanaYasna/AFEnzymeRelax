#!/bin/bash -l
#SBATCH --job-name=dim_red
#SBATCH --output=reductions.txt
#SBATCH --partition sixhour
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=10G
#SBATCH --time=6:00:00

module load conda
conda activate graph
cd /kuhpc/work/slusky/syasna_sta/func_pred/code/dimenet_focal_loss
export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' 
export 'PYTORCH_NO_CUDA_MEMORY_CACHING=1'
conda run -n graph python red_combos.py