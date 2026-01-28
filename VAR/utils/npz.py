import os, glob
import numpy as np
from tqdm import tqdm
from PIL import Image
import logging
logger = logging.getLogger(__name__)

def create_npz_from_sample_folder(sample_folder: str, var_model_size: str, output: str, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples. Refer to DiT.
    """
    samples = []
    pngs = glob.glob(os.path.join(sample_folder, '*.png')) + glob.glob(os.path.join(sample_folder, '*.PNG'))
    assert len(pngs) == num, f'{len(pngs)} png files found in {sample_folder}, but expected {num}'
    for png in tqdm(pngs, desc='Building .npz file from samples (png only)', disable=True):
        with Image.open(png) as sample_pil:
            sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f'{output}/{var_model_size}.npz'
    np.savez(npz_path, arr_0=samples)
    logger.info(f'Saved .npz file to {npz_path} [shape={samples.shape}].')
    return npz_path