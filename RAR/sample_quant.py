import sys, argparse
sys.path.append("./")
import os
import logging, datetime
import torch
from PIL import Image
import numpy as np
import demo_util
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from utils.train_utils import create_pretrained_tokenizer
from quant import *
from utils.npz import create_npz_from_sample_folder
logger = logging.getLogger(__name__)


def get_args_parser():
    parser = argparse.ArgumentParser(description="RAR", add_help=False)
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--n_batch", default=100, type=int)
    parser.add_argument("--n_sample", default=50000, type=int)
    parser.add_argument("--quant_act", action="store_true", default=True)
    parser.add_argument("--quant_wei", action="store_true", default=True)
    parser.add_argument('--w_bits', default=6, type=int, help='bit-precision of weights')
    parser.add_argument('--a_bits', default=6, type=int, help='bit-precision of activation')
    parser.add_argument("--sample", action='store_true', default=True)
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--model", type=str, default="rar_b", choices=["rar_b", "rar_l", "rar_xl", "rar_xxl"])
    parser.add_argument("--ckpt_dir", type=str, required=True, help='Path to the model weight file')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RAR', parents=[get_args_parser()])
    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = os.path.join(args.output, "log", now+'-RAR_quant')
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

    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    q_model = quant_model(generator, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.to(device)
    q_model.eval()

    # Initial quantization
    input_ids, condition = torch.load(f"{args.output}/cali_{rar_model_size}_128_dgc.pth")

    logger.info(75 * "*")
    logger.info("calibration smooth factor first")
    set_smooth_quantize_params(q_model, input_ids, condition)

    logger.info("Performing initial quantization ...")
    set_quant_state(q_model, input_quant=True, weight_quant=True)

    with torch.no_grad():
        _ = q_model.forward_fn(input_ids, condition, is_sampling=True)

    # Shifting the activation channel to 0 symmetry eliminates the effect of uneven ||X||^2 distribution on scaling gains
    shift_activation_to_sym(q_model)
    logger.info(f"reset_quantize_params")
    with torch.no_grad():
        _ = q_model.forward_fn(input_ids, condition, is_sampling=True)

    logger.info(75 * "*")
    logger.info("start gps: calculating optimal factor")
    set_calculating_optimal_factor(q_model, input_ids, condition)

    # Absorb scaling factor
    absorb_gps_scaling_factor(q_model)

    logger.info(f"reset_quantize_params")
    set_quant_state(q_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = q_model.forward_fn(input_ids, condition, is_sampling=True)

    if args.sample:
        logger.info("Start sample ...")
        seed_everything(args.seed)
        base_count = 0
        n_batch = args.n_batch
        all_classes = list(range(1000)) * int(args.n_sample / 1000)
        sample_folder_dir = f"{args.output}/{rar_model_size}_image"
        os.makedirs(sample_folder_dir, exist_ok=True)

        iterator = tqdm(range(len(all_classes)//n_batch), desc='Sampler')
        for i, class_num in enumerate(iterator):
            seed_everything(args.seed+i)
            sample_labels = torch.tensor(all_classes[i*n_batch:(i+1)*n_batch]).to(device)
            samples = demo_util.sample_fn(
                generator=generator,
                tokenizer=tokenizer,
                labels=sample_labels.long(),
                randomize_temperature=config.model.generator.randomize_temperature,
                guidance_scale=config.model.generator.guidance_scale,
                guidance_scale_pow=config.model.generator.guidance_scale_pow,
                device=device,
                kv_cache=True
            )
            for i, sample in enumerate(samples):
                Image.fromarray(sample).save(f"{sample_folder_dir}/{base_count:06d}.png")
                base_count += 1

        logger.info("Trans image to .npz ...")
        create_npz_from_sample_folder(sample_folder_dir, rar_model_size, args.output, num=base_count)
    logger.info("down!")
