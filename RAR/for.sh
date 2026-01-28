

source ~/.bashrc

# PTQ
source ~/miniconda3/etc/profile.d/conda.sh
conda activate arvg
echo "The environment arvg has been activated."


CUDA_VISIBLE_DEVICES=0 python ./sample_quant.py --sample --quant_wei --quant_act --n_batch 100 --n_sample 50000 --w_bits 6 --a_bits 6

# test
CUDA_VISIBLE_DEVICES=0 python ./evaluation/test.py /data/VIRTUAL_imagenet256_labeled.npz ./output/rar_b.npz
# CUDA_VISIBLE_DEVICES=0 python ./evaluation/test.py /data/VIRTUAL_imagenet256_labeled.npz ./output/rar_l.npz
# CUDA_VISIBLE_DEVICES=0 python ./evaluation/test.py /data/VIRTUAL_imagenet256_labeled.npz ./output/rar_xl.npz
# CUDA_VISIBLE_DEVICES=0 python ./evaluation/test.py /data/VIRTUAL_imagenet256_labeled.npz ./output/rar_xxl.npz
conda deactivate

