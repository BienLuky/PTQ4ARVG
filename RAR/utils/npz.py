import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

def create_npz_from_sample_folder(sample_dir, rar_model_size, output, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    logger.info("Building .npz file from samples")
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples", disable=True):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{output}/{rar_model_size}.npz"
    np.savez(npz_path, arr_0=samples)
    logger.info(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path
