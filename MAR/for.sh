source ~/.bashrc

#PTQ
source ~/miniconda3/etc/profile.d/conda.sh
conda activate arvg
echo "The environment arvg has been activated."


CUDA_VISIBLE_DEVICES=0 python main_mar_quant.py --model_name mar_b --num_images 50000 --eval_bsz 256 --w_bits 6 --a_bits 6

# test
CUDA_VISIBLE_DEVICES=0 python ./evaluation/test.py ./VIRTUAL_imagenet256_labeled.npz ./output/mar_b.npz


