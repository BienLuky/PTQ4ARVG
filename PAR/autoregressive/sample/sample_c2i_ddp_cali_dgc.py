import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os, sys, logging, datetime
sys.path.append("./")
import argparse

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
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


def main(args):
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = os.path.join(args.output, "log", now+'-PAR_cali_dgc')
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

    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    device = 'cuda'
    seed_everything(args.seed)

    model_id = {'PAR-XL-4x':0, 'PAR-XXL-4x':1, 'PAR-3B-4x':2, 'PAR-3B-16x':3}[args.model]
    gpt = ["PAR-XL-4x", "PAR-XXL-4x", "PAR-3B-4x", "PAR-3B-16x"][model_id]
    args.gpt_ckpt = f"{args.ckpt_dir}/{gpt}.pt"
    args.vq_ckpt = f"{args.ckpt_dir}/vq_ds16_c2i.pt"
    args.gpt_model = ["GPT-XL", "GPT-XXL", "GPT-3B", "GPT-3B"][model_id]
    args.spe_token_num = [3, 3, 3, 15][model_id]
    args.ar_token_num = [4, 4, 4, 16][model_id]
    args.cfg_scale = [1.5, 1.435, 1.345, 1.5][model_id]

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        spe_token_num=args.spe_token_num,
        ar_token_num=args.ar_token_num,
    ).to(device=device, dtype=precision)
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp: # fsdp
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")

    gpt_model.load_state_dict(model_weight, strict=True)
    gpt_model.eval()
    del checkpoint

    number = args.cali_num_samples
    batch = args.cali_batch
    hooks = []
    hooks.append(AttentionMap(gpt_model.layers[-1]))

    hs, freqs_cis, input_pos, mask = torch.load(f"{args.output}/cali_{gpt}_{number}.pth")
    hs = torch.cat(hs, dim=1).to(device)
    freqs_cis, input_pos, mask = freqs_cis.to(device), input_pos.to(device), mask.to(device)

    with torch.no_grad():
        _ = gpt_model.forward_fn(hs, freqs_cis, input_pos, mask)
    samples = hooks[0].out

    cond_samples = []
    uncond_samples = []
    for i in range(int(hs.size(0)/(batch*2))):
        idx_cond = torch.tensor(range(batch)) + (2*i) * batch
        idx_uncond = torch.tensor(range(batch)) + (2*i+1) * batch
        cond_samples.append(samples[idx_cond])
        uncond_samples.append(samples[idx_uncond])
    samples = torch.cat(cond_samples+uncond_samples)

    dist_cond = mahalanobis_to_distribution(samples[:number].to(dtype=torch.float32))
    dist_uncond = mahalanobis_to_distribution(samples[number:].to(dtype=torch.float32))
    if torch.allclose(dist_cond, dist_uncond, atol=1e-03): #Due to the small amount of data involved in the calculation, the default difference in the amount of information is not significant as the amount of information is equal
        pct = 0.5
    else:
        dist = torch.cat([dist_cond, dist_uncond])
        _, topk_indices = torch.topk(dist, number)
        pct = (topk_indices < number).sum().item() / number
    logger.info(f"pct: {pct}")
    num_cond = round(pct*batch)
    num_uncond = batch - num_cond

    idx = []
    for i in range(int(hs.size(0)/(batch*2))):
        idx_cond = torch.tensor(range(num_cond)) + (2*i) * batch
        idx_uncond = torch.tensor(range(num_uncond)) + (2*i+1) * batch
        idx.append(torch.cat([idx_cond, idx_uncond]))
    idx = torch.cat(idx).long()

    hs, freqs_cis, input_pos, mask = torch.load(f"{args.output}/cali_{gpt}_{number}.pth")
    for i in range(len(hs)):
        hs[i] = hs[i][idx]
    mask = mask[idx]
    torch.save((hs, freqs_cis, input_pos, mask), f"{args.output}/cali_{gpt}_{number}_dgc.pth")
    logger.info(f"cali num: {mask.size(0)}")

    logger.info("down!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default="")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--spe-token-num", type=int, default=3, help="number of learnable tokens")
    parser.add_argument("--ar-token-num", type=int, default=4, help="number of parallel prediction tokens")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=True)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default="/vq_ds16_c2i.pt", help="none")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=5000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")

    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument('--cali_num_samples', type=int, default=128) 
    parser.add_argument('--cali_batch', type=int, default=32) 
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--model", type=str, default="PAR-XL-4x", choices=["PAR-XL-4x", "PAR-XXL-4x", "PAR-3B-4x", "PAR-3B-16x"])
    parser.add_argument("--ckpt_dir", type=str, required=True, help='Path to the model weight file')
    args = parser.parse_args()
    main(args)