import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
from tqdm import tqdm
import os, sys, logging, datetime
sys.path.append("./")
from PIL import Image
import argparse

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
from quant.utils import seed_everything, create_npz_from_sample_folder
logger = logging.getLogger(__name__)


def main(args):
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = os.path.join(args.output, "log", now+'-PAR_sample')
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

    # Create folder to save samples:
    sample_folder_dir = f"{args.output}/{gpt}_image"
    os.makedirs(sample_folder_dir, exist_ok=True)
    logger.info(f"Saving .png samples at {sample_folder_dir}")

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    total_samples = args.num_fid_samples
    logger.info(f"Total number of images that will be sampled: {total_samples}")

    iterations = int(total_samples // args.batch_size)
    pbar = tqdm(range(iterations)) 
    base_count = 0
    for _ in pbar:
        # Sample inputs:
        c_indices = torch.randint(0, args.num_classes, (args.batch_size,), device=device)
        qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]

        index_sample = generate(
            gpt_model, c_indices, latent_size ** 2,
            cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True, 
            spe_token_num=args.spe_token_num,
            ar_token_num=args.ar_token_num,
            )
        ar_token_num = args.ar_token_num
        sub_num = int(ar_token_num**0.5)
        index_sample = index_sample.reshape(index_sample.shape[0],latent_size//sub_num, latent_size//sub_num, sub_num, sub_num)
        index_sample = index_sample.permute(0, 3, 1, 4, 2)
        index_sample = index_sample.reshape(index_sample.shape[0], latent_size, latent_size)
        index_sample = index_sample.reshape(index_sample.shape[0],-1)
        samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
        if args.image_size_eval != args.image_size:
            samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        
        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            Image.fromarray(sample).save(f"{sample_folder_dir}/{base_count:06d}.png")
            base_count += 1

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    logger.info("Trans image to .npz ...")
    create_npz_from_sample_folder(sample_folder_dir, gpt, args.output, args.num_fid_samples)
    logger.info("Done.")


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
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")

    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--model", type=str, default="PAR-XL-4x", choices=["PAR-XL-4x", "PAR-XXL-4x", "PAR-3B-4x", "PAR-3B-16x"])
    parser.add_argument("--ckpt_dir", type=str, required=True, help='Path to the model weight file')
    args = parser.parse_args()
    main(args)