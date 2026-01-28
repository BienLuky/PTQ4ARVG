import os, logging, argparse, datetime
import os.path as osp
import torch, torchvision
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from models import VQVAE, build_vae_var
from utils.npz import create_npz_from_sample_folder
from quant import *
from models.basic_var import SelfAttention
from quant.set_quantize_params import *
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    sample_folder_dir = f"/home/lxw/PTQ4ARVG/VAR/output/var_d16_image"
    var_model_size = "var_d16"
    output = "/home/lxw/PTQ4ARVG/VAR/output"

    logger.info("Trans image to .npz ...")
    create_npz_from_sample_folder(sample_folder_dir, var_model_size, output, num=5000)
    logger.info("down!")