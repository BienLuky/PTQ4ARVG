import os, logging, argparse, datetime
import torch, torchvision
from models import VQVAE, build_vae_var
from quant.utils import seed_everything
from models.basic_var import SelfAttention
from torch.nn import Parameter
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
    parser.add_argument('--cali_batch', type=int, default=32) 
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--model", type=str, default="var_d16", choices=["var_d16", "var_d20", "var_d24", "var_d30"])
    parser.add_argument("--ckpt_dir", type=str, required=True, help='Path to the model weight file')
    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = os.path.join(args.output, "log", now+'-VAR_cali_dgc')
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
    MODEL_DEPTH = {'var_d16':16, 'var_d20':20, 'var_d24':24, 'var_d30':30}[args.model]
    assert MODEL_DEPTH in {16, 20, 24, 30}
    var_model_size = f"var_d{MODEL_DEPTH}"

    # download checkpoint
    vae_ckpt = f"{args.ckpt_dir}/vae_ch160v4096z32.pth"
    var_ckpt = f"{args.ckpt_dir}/{var_model_size}.pth"

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'vae' not in globals() or 'var' not in globals():
        vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
            device=device, patch_nums=patch_nums,
            num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
        )

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    for name, m in var.named_modules():
        if isinstance(m, (SelfAttention)):
            assert m.mat_qkv.bias == None
            bias = torch.cat((m.q_bias, m.zero_k_bias, m.v_bias))
            m.mat_qkv.bias = Parameter(torch.Tensor(bias.size(0)))
            m.mat_qkv.bias.data = bias

    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    logger.info(f'prepare finished.')

    cfg = 1.5 #@param {type:"slider", min:1, max:10, step:0.1}
    top_p = 0.96
    top_k = 900
    more_smooth = False # True for more smooth output

    # seed
    seed_everything(args.seed)

    number = args.cali_num_samples
    batch = args.cali_batch
    hooks = []
    hooks.append(AttentionMap(var.blocks[-1]))

    inputs, conds = torch.load(f"{args.output}/cali_{var_model_size}_{number}.pth")
    token_map = torch.cat(inputs, dim=1).to(device)
    cond_BD_or_gss = conds.to(device)

    with torch.no_grad():
        _ = var.forward_fn(token_map, cond_BD_or_gss)
    samples = hooks[0].out

    cond_samples = []
    uncond_samples = []
    for i in range(int(token_map.size(0)/(batch*2))):
        idx_cond = torch.tensor(range(batch)) + (2*i) * batch
        idx_uncond = torch.tensor(range(batch)) + (2*i+1) * batch
        cond_samples.append(samples[idx_cond])
        uncond_samples.append(samples[idx_uncond])
    samples = torch.cat(cond_samples+uncond_samples)

    dist_cond = mahalanobis_to_distribution(samples[:number])
    dist_uncond = mahalanobis_to_distribution(samples[number:])
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
    for i in range(int(token_map.size(0)/(batch*2))):
        idx_cond = torch.tensor(range(num_cond)) + (2*i) * batch
        idx_uncond = torch.tensor(range(num_uncond)) + (2*i+1) * batch
        idx.append(torch.cat([idx_cond, idx_uncond]))
    idx = torch.cat(idx)

    for i in range(len(inputs)):
        inputs[i] = inputs[i][idx]
    conds = conds[idx]
    torch.save((inputs, conds), f"{args.output}/cali_{var_model_size}_{number}_dgc.pth")
    logger.info(f"cali num: {conds.size(0)}")

    logger.info("down!")