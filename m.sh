bash s3mount.sh

cd LLaMA-Factory



python merge_tokenized_datasets.py \
    saves/videochat-o3-sft-3m/dataset/sft saves/videochat-o3-sft-30s/dataset/sft /mnt/petrelfs/zhangzhiqiu/LLaMA-Factory/saves/videochat-o3-sft-2m/dataset/sft \
    --output_path /mnt/petrelfs/zhangzhiqiu/sft_data/preprocessed_data/sft \
    --mix_strategy concat