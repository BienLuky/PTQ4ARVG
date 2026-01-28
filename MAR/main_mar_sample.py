import numpy as np
import os, sys
sys.path.append("./")
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import util.misc as misc

from models.vae import AutoencoderKL
from models import mar
from engine_mar import evaluate
import copy
from quant.utils import seed_everything, get_args_parser


def main(args):
    seed_everything(42)
    model_id = {'mar_b':0, 'mar_l':1, 'mar_xl':2}[args.model_name]
    model_size = ["mar_b", "mar_l", "mar_xl"][model_id]
    args.model = ["mar_base", "mar_large", "mar_huge"][model_id]
    args.diffloss_d = [6, 8, 12][model_id]
    args.diffloss_w = [1024, 1280, 1536][model_id]
    args.cfg = [2.9, 3.0, 3.2][model_id]
    ckpt_dir = ["pretrained_models/mar/mar_base", "pretrained_models/mar/mar_large", "pretrained_models/mar/mar_huge"][model_id]
    vae_path = os.path.join(args.resume, args.vae_path)
    ckpt_dir = os.path.join(args.resume, ckpt_dir)
    sample_folder_dir = f"{args.output_dir}/{model_size}_image"
    os.makedirs(sample_folder_dir, exist_ok=True)

    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    # define the vae and mar model
    vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=vae_path).cuda().eval()
    for param in vae.parameters():
        param.requires_grad = False

    model = mar.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        mask_ratio_min=args.mask_ratio_min,
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        grad_checkpointing=args.grad_checkpointing,
    )

    print("Model = %s" % str(model))
    for param in model.parameters():
        param.requires_grad = False

    model.to(device)
    model_without_ddp = model

    # resume training
    if args.resume and os.path.exists(os.path.join(ckpt_dir, "checkpoint-last.pth")):
        checkpoint = torch.load(os.path.join(ckpt_dir, "checkpoint-last.pth"), map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        model_params = list(model_without_ddp.parameters())
        ema_state_dict = checkpoint['model_ema']
        ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resume checkpoint %s" % ckpt_dir)
        del checkpoint

    model_without_ddp.eval()
    use_ema = True
    # switch to ema params
    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    save_folder = sample_folder_dir
    # evaluate FID and IS
    if args.evaluate:
        torch.cuda.empty_cache()
        evaluate(model_without_ddp, vae, model_size, args, 0, batch_size=args.eval_bsz, log_writer=log_writer,
                 cfg=args.cfg, use_ema=True, save_folder=save_folder, output_dir=args.output_dir)
        return


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
