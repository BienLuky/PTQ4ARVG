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


def get_args_parser():
    parser = argparse.ArgumentParser(description="VAR", add_help=False)
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--n_batch", default=100, type=int)
    parser.add_argument("--n_sample", default=50000, type=int)
    parser.add_argument("--quant_act", action="store_true", default=True)
    parser.add_argument("--quant_wei", action="store_true", default=True)
    parser.add_argument('--w_bits', default=6,type=int, help='bit-precision of weights')
    parser.add_argument('--a_bits', default=6,type=int, help='bit-precision of activation')
    parser.add_argument("--sample", action='store_true', default=True)
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--model", type=str, default="var_d16", choices=["var_d16", "var_d20", "var_d24", "var_d30"])
    parser.add_argument("--ckpt_dir", type=str, required=True, help='Path to the model weight file')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VAR', parents=[get_args_parser()])
    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = os.path.join(args.output, "log", now+'-VAR_quant')
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

    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False, 'leaf_param':True}
    q_model = quant_model(var, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.to(device)
    q_model.eval()

    inputs, conds = torch.load(f"{args.output}/cali_{var_model_size}_128_dgc.pth")
    token_maps = []
    for cali in inputs:
        next_token_map = cali
        token_maps.append(next_token_map)
    token_map = torch.cat(token_maps, dim=1).to(device)
    cond_BD_or_gss = conds.to(device)

    batch = 64
    logger.info(75 * "*")
    logger.info("calibration smooth factor first")
    set_smooth_quantize_params(q_model, token_map, cond_BD_or_gss, batch_size=batch)

    logger.info("Performing initial quantization ...")
    set_weight_quantize_params(q_model, token_map, cond_BD_or_gss)
    set_act_quantize_params(q_model, token_map, cond_BD_or_gss, batch_size=batch)
    set_quant_state(q_model, input_quant=True, weight_quant=True)

    # Shifting the activation channel to 0 symmetry eliminates the effect of uneven ||X||^2 distribution on scaling gains
    shift_activation_to_sym(q_model)

    reset_weight_quantize_params(q_model, token_map, cond_BD_or_gss)
    reset_act_quantize_params(q_model, token_map, cond_BD_or_gss, batch_size=batch)

    logger.info(75 * "*")
    logger.info("calculating optimal factor")
    set_calculating_optimal_factor(q_model, token_map, cond_BD_or_gss, batch_size=batch)

    # Absorb scaling factor
    absorb_gps_scaling_factor(q_model)

    reset_weight_quantize_params(q_model, token_map, cond_BD_or_gss)
    set_act_quantize_params(q_model, token_map, cond_BD_or_gss, batch_size=batch)
    set_quant_state(q_model, input_quant=True, weight_quant=True)

    if args.sample:
        logger.info("Start sample ...")
        base_count = 0
        n_batch = args.n_batch
        class_nums, per_class = 1000, int(args.n_sample/1000)
        all_classes = [label for label in range(class_nums) for _ in range(per_class)]
        sample_folder_dir = f"{args.output}/{var_model_size}_image"
        os.makedirs(sample_folder_dir, exist_ok=True)

        iterator = tqdm(range(len(all_classes)//n_batch), desc='Sampler')
        with torch.inference_mode():
            with torch.autocast('cuda', dtype=torch.float16, enabled=True, cache_enabled=True):    # using bfloat16 can be faster # , dtype=torch.float16, dtype=torch.float32
                for i, class_num in enumerate(iterator):
                    sample_labels = torch.tensor(all_classes[i*n_batch:(i+1)*n_batch]).to(device)
                    B = len(sample_labels)
                    label_B: torch.LongTensor = torch.tensor(sample_labels, device=device)
                    recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=top_k, top_p=top_p, g_seed=args.seed, more_smooth=more_smooth)
                    for i in range(recon_B3HW.size(0)):
                        img = recon_B3HW[i].cpu().permute(1, 2, 0).mul_(255).byte().numpy()
                        Image.fromarray(img).save(f"{sample_folder_dir}/{base_count:06d}.png")
                        base_count += 1

        logger.info("Trans image to .npz ...")
        create_npz_from_sample_folder(sample_folder_dir, var_model_size, args.output, num=base_count)
    logger.info("down!")