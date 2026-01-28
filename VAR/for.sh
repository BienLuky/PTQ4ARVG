source ~/.bashrc

# PTQ
source ~/miniconda3/etc/profile.d/conda.sh
conda activate arvg
echo "The environment arvg has been activated."


CUDA_VISIBLE_DEVICES=0 python ./sample_quant.py --sample --quant_wei --quant_act --n_batch 100 --w_bits 6 --a_bits 6

# test
CUDA_VISIBLE_DEVICES=0 python ./evaluation/test.py /data/VIRTUAL_imagenet256_labeled.npz ./output/var_d16.npz
# CUDA_VISIBLE_DEVICES=0 python ./evaluation/test.py /data/VIRTUAL_imagenet256_labeled.npz ./output/var_d20.npz
# CUDA_VISIBLE_DEVICES=0 python ./evaluation/test.py /data/VIRTUAL_imagenet256_labeled.npz ./output/var_d24.npz
# CUDA_VISIBLE_DEVICES=0 python ./evaluation/test.py /data/VIRTUAL_imagenet256_labeled.npz ./output/var_d30.npz
conda deactivate
