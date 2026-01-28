import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
logger = logging.getLogger(__name__)

class AttentionMap:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        # self.ac_output = []

    def hook_fn(self, module, input, output):
        self.out = output
        self.feature = input
        # self.ac_output.append(output)

    def remove(self):
        self.hook.remove()


def seed_everything(seed):
    '''
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_npz_from_sample_folder(sample_dir, name, output, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{output}/{name}.npz"
    np.savez(npz_path, arr_0=samples)
    logger.info(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path
