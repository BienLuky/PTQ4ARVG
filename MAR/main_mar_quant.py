import logging
import os, sys
sys.path.append("./")
from pathlib import Path

import torch
from torch.nn import Parameter
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc

from models.vae import AutoencoderKL
from models import mar
from engine_mar import evaluate
import copy
from quant.utils import seed_everything, get_args_parser
from quant.quant_model import quant_model, set_quant_state
from quant.set_quantize_params import set_smooth_quantize_params, set_calculating_optimal_factor


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

    log_path = os.path.join(args.output_dir, "run.log")
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

    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed_everything(args.seed)
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

    logger.info("Model = %s" % str(model))
    for param in model.parameters():
        param.requires_grad = False

    model.to(device)
    model_without_ddp = model

    if args.resume and os.path.exists(os.path.join(ckpt_dir, "checkpoint-last.pth")):
        checkpoint = torch.load(os.path.join(ckpt_dir, "checkpoint-last.pth"), map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        model_params = list(model_without_ddp.parameters())
        ema_state_dict = checkpoint['model_ema']
        ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        logger.info("Resume checkpoint %s" % ckpt_dir)
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

    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    q_model = quant_model(model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.to(device)
    q_model.eval()

    # Initial quantization
    number = args.cali_num_samples
    tokens, mask, class_embedding = torch.load(f"{args.output_dir}/cali_{model_size}_{number}_dgc.pth")
  
    logger.info(75 * "*")
    logger.info("calibration smooth factor first")
    set_smooth_quantize_params(q_model, tokens, mask, class_embedding)

    logger.info("Performing initial quantization ...")
    set_quant_state(q_model, input_quant=True, weight_quant=True)

    with torch.no_grad():
        _ = q_model.forward_fn(tokens, mask, class_embedding)

    # '''
    with torch.no_grad():
        module_dict={}
        for name, module in q_model.named_modules():
            module_dict[name] = module
            idx = name.rfind('.')
            if idx == -1:
                idx = 0
            father_name = name[:idx]
            module_name = name[idx:]
            if father_name in module_dict:
                father_module = module_dict[father_name]
            else:
                raise RuntimeError(f"father module {father_name} not found")

            if 'norm1' in module_name or 'norm2' in module_name and 'diffloss' not in name:
                logger.info(name)
                adaLN_module = module
                if 'norm1' in name:
                    next_module = father_module.attn.qkv
                elif 'norm2' in name:
                    next_module = father_module.mlp.fc1

                for m in [next_module]:
                    r, b, m.input_quantizer.delta, m.input_quantizer.zero_point = \
                        m.SmoothModule.init_sym_shift(m.input_quantizer.delta, m.input_quantizer.zero_point, m.input_quantizer.n_bits)

                    if m.bias is not None:
                        m.bias.data = m.bias.data + torch.mm(m.weight.data, b.reshape(-1,1)).reshape(-1)
                    else:
                        m.bias = Parameter(torch.Tensor(m.out_features))
                        m.bias.data = torch.mm(m.weight.data, b.reshape(-1,1)).reshape(-1)

                    m.input_quantizer.channel_wise = True
                    m.input_quantizer.inited = False

                # absorb
                p1 = next_module.SmoothModule.shifts
                s1 = next_module.SmoothModule.scales
                ps1 = p1 * s1

                add_bias = torch.cat([-ps1])
                scale = torch.cat([s1])

                adaLN_module.weight.data = adaLN_module.weight.data / scale
                adaLN_module.bias.data = (adaLN_module.bias.data + add_bias) / scale

                next_module.SmoothModule.scales = None
                next_module.SmoothModule.shifts = None

    # ''' 
    logger.info(f"reset_quantize_params")
    with torch.no_grad():
        _ = q_model.forward_fn(tokens, mask, class_embedding)

    logger.info(75 * "*")
    logger.info("calculating optimal factor")
    set_calculating_optimal_factor(q_model, tokens, mask, class_embedding)

    # '''
    with torch.no_grad():
        module_dict={}
        for name, module in q_model.named_modules():
            module_dict[name] = module
            idx = name.rfind('.')
            if idx == -1:
                idx = 0
            father_name = name[:idx]
            module_name = name[idx:]
            if father_name in module_dict:
                father_module = module_dict[father_name]
            else:
                raise RuntimeError(f"father module {father_name} not found")

            if 'norm1' in module_name or 'norm2' in module_name and 'diffloss' not in name:
                logger.info(name)
                adaLN_module = module
                if 'norm1' in name:
                    next_module = father_module.attn.qkv
                elif 'norm2' in name:
                    next_module = father_module.mlp.fc1

                for m in [next_module]:

                    from quant.quantizer import TokenAwareUniformQuantizer
                    m.input_quantizer = TokenAwareUniformQuantizer(**m.input_quant_params)
                    m.input_quantizer.channel_wise = False
                    m.input_quantizer.inited = False
                    m.weight_quantizer.inited = False
                    m.sink_aware = False
                # absorb
                p1 = next_module.SmoothModule.shifts
                s1 = next_module.SmoothModule.scales
                ps1 = p1 * s1

                add_bias = torch.cat([-ps1])
                scale = torch.cat([s1])
                adaLN_module.weight.data = adaLN_module.weight.data / scale
                adaLN_module.bias.data = (adaLN_module.bias.data + add_bias) / scale

                next_module.SmoothModule.scales = None
                next_module.SmoothModule.shifts = None


    logger.info(f"reset_quantize_params")
    set_quant_state(q_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = q_model.forward_fn(tokens, mask, class_embedding)

    # evaluate FID and IS
    save_folder = sample_folder_dir
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
