import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
import random
import glob, os
logger = logging.getLogger(__name__)


def create_npz_from_sample_folder(sample_folder: str, mar_model_size: str, output: str, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples. Refer to DiT.
    """

    samples = []
    pngs = glob.glob(os.path.join(sample_folder, '*.png')) + glob.glob(os.path.join(sample_folder, '*.PNG'))
    print(f'Imageflor have {len(pngs)} .png files')
    pngs = random.sample(pngs, num)
    for png in tqdm(pngs, desc='Building .npz file from samples (png only)', disable=True):
        with Image.open(png) as sample_pil:
            sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f'{output}/{mar_model_size}.npz'
    np.savez(npz_path, arr_0=samples)
    print(f'Saved .npz file to {npz_path} [shape={samples.shape}].')
    return npz_path