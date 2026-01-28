import sys, argparse
sys.path.append("./")
import os
import logging, datetime
import torch
import demo_util
from huggingface_hub import hf_hub_download
from utils.train_utils import create_pretrained_tokenizer
from quant.utils import seed_everything
from quant import *
logger = logging.getLogger(__name__)


def mahalanobis_to_distribution(samples):
    samples = samples[:,0,:]
    samples = samples.view(samples.size(0), -1)  

    # calculate mahalanobis
    mean = samples.mean(dim=0) 
    X_centered = samples - mean 
    cov = (X_centered.T @ X_centered) / (samples.size(0) - 1) 
    cov_inv = torch.linalg.pinv(cov) 

    dists = torch.zeros((samples.size(0),))
    for i in range(samples.size(0)):
        x = samples[i]
        diff = x - mean 
        dist = torch.sqrt(diff @ cov_inv @ diff.T) 
        dists[i] = dist
    return dists


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument('--cali_num_samples', type=int, default=128) 
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--model", type=str, default="rar_b", choices=["rar_b", "rar_l", "rar_xl", "rar_xxl"])
    parser.add_argument("--ckpt_dir", type=str, required=True, help='Path to the model weight file')
    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = os.path.join(args.output, "log", now+'-RAR_cali_dgc')
    os.makedirs(logdir)
    log_path = os.path.join(logdir, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(75 * "=")
    logger.info(args)

    seed_everything(args.seed)
    # Choose one from ["rar_b_imagenet", "rar_l_imagenet", "rar_xl_imagenet", "rar_xxl_imagenet"]
    rar_model_size = args.model

    # download the maskgit-vq tokenizer
    hf_hub_download(repo_id="fun-research/TiTok", filename=f"maskgit-vqgan-imagenet-f16-256.bin", local_dir=args.ckpt_dir)
    # download the rar generator weight
    hf_hub_download(repo_id="yucornetto/RAR", filename=f"{rar_model_size}.bin", local_dir=args.ckpt_dir)

    # load config
    config = demo_util.get_config("./configs/training/generator/rar.yaml")
    config.experiment.generator_checkpoint = f"{args.ckpt_dir}/{rar_model_size}.bin"
    config.model.vq_model.pretrained_tokenizer_weight = f"{args.ckpt_dir}/maskgit-vqgan-imagenet-f16-256.bin"
    config.model.generator.hidden_size = {"rar_b": 768, "rar_l": 1024, "rar_xl": 1280, "rar_xxl": 1408}[rar_model_size]
    config.model.generator.num_hidden_layers = {"rar_b": 24, "rar_l": 24, "rar_xl": 32, "rar_xxl": 40}[rar_model_size]
    config.model.generator.num_attention_heads = 16
    config.model.generator.intermediate_size = {"rar_b": 3072, "rar_l": 4096, "rar_xl": 5120, "rar_xxl": 6144}[rar_model_size]
    config.model.generator.randomize_temperature = {"rar_b": 1.0, "rar_l": 1.02, "rar_xl": 1.02, "rar_xxl": 1.02}[rar_model_size]
    config.model.generator.guidance_scale = {"rar_b": 16.0, "rar_l": 15.5, "rar_xl": 6.9, "rar_xxl": 8.0}[rar_model_size]
    config.model.generator.guidance_scale_pow = {"rar_b": 2.75, "rar_l": 2.5, "rar_xl": 1.5, "rar_xxl": 1.2}[rar_model_size]
    device = "cuda"
    # maskgit-vq as tokenizer
    tokenizer = create_pretrained_tokenizer(config)
    generator = demo_util.get_rar_generator(config)
    tokenizer.to(device)
    generator.to(device)

    number = args.cali_num_samples
    hooks = []
    hooks.append(AttentionMap(generator.blocks[-1]))

    # Initial quantization
    input_ids, condition = torch.load(f"{args.output}/cali_{rar_model_size}_{number}.pth")

    with torch.no_grad():
        _ = generator.forward_fn(input_ids, condition, is_sampling=True)
    samples = hooks[0].out

    dist_cond = mahalanobis_to_distribution(samples[:number])
    dist_uncond = mahalanobis_to_distribution(samples[number:])
    if torch.allclose(dist_cond, dist_uncond, atol=1e-03): #Due to the small amount of data involved in the calculation, the default difference in the amount of information is not significant as the amount of information is equal
        pct = 0.5
    else:
        dist = torch.cat([dist_cond, dist_uncond])
        _, topk_indices = torch.topk(dist, number)
        pct = (topk_indices < number).sum().item() / number
    num_cond = round(pct*number)
    num_uncond = number - num_cond

    idx_cond = torch.tensor(range(num_cond)) 
    idx_uncond = torch.tensor(range(num_uncond)) + number
    idx = torch.cat([idx_cond, idx_uncond]).long()

    input_ids, condition = input_ids[idx], condition[idx]
    torch.save((input_ids, condition), f"{args.output}/cali_{rar_model_size}_{number}_dgc.pth")
    logger.info(f"cali num: {condition.size(0)}")

    logger.info("down!")