source ~/.bashrc

# PTQ
source ~/miniconda3/etc/profile.d/conda.sh
conda activate arvg
echo "The environment arvg has been activated."


CUDA_VISIBLE_DEVICES=0 python ./autoregressive/sample/sample_c2i_ddp_quant.py \
    --num-fid-samples 50000 --batch-size 50 --w_bits 6 --a_bits 6

# test
CUDA_VISIBLE_DEVICES=0 python ./evaluation/test.py /data/VIRTUAL_imagenet256_labeled.npz ./output/PAR-XL-4x.npz

