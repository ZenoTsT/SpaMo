#!/bin/bash
#SBATCH --job-name=vit_h2s_s2_s7
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=tesi_ztesta
#SBATCH --partition=all_usr_prod

module load anaconda3/2023.09-0-none-none
source activate spamo
cd /homes/ztesta/SpaMo || exit 1

python scripts/vit_extract_feature.py   --anno_root preprocess/How2Sign   --video_root /work/tesi_ztesta/How2Sign   --device cuda:0   --batch_size 8   --cache_dir /work/tesi_ztesta/models_cache   --save_dir /work/tesi_ztesta/features/how2sign/vit_feat_s2   --model_name openai/clip-vit-large-patch14   --s2_mode s2wrapping   --scales 1 2   --num_shards 8   --shard_id 7
