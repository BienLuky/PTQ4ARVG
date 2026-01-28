'''
First, remember to uncomment line 477-480 in ./modeling/rar.py and comment them after finish collecting.
'''
import sys, argparse
sys.path.append("./")
import os
import logging, datetime
import torch
import demo_util
from huggingface_hub import hf_hub_download
from utils.train_utils import create_pretrained_tokenizer
from quant.utils import seed_everything
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument('--cali_num_samples', type=int, default=128) 
    parser.add_argument('--cali_batch', type=int, default=128) 
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--model", type=str, default="rar_b", choices=["rar_b", "rar_l", "rar_xl", "rar_xxl"])
    parser.add_argument("--ckpt_dir", type=str, required=True, help='Path to the model weight file')
    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = os.path.join(args.output, "log", now+'-RAR_cali')
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
    logger.info(f"cali number: {number}")
    with torch.no_grad():
    # generate an image
        sample_labels = list(torch.randint(0, 999, size=(number,))) # random IN-1k class
        generated_image = demo_util.sample_fn(
            generator=generator,
            tokenizer=tokenizer,
            labels=sample_labels,
            randomize_temperature=config.model.generator.randomize_temperature,
            guidance_scale=config.model.generator.guidance_scale,
            guidance_scale_pow=config.model.generator.guidance_scale_pow,
            device=device,
            kv_cache=True
        )
    '''
    First, remember to uncomment line 477-480 in ./RAR/modeling/rar.py and comment them after finish collecting.
    '''
    import quant.globalvar as globalvar   
    input_list = globalvar.getInputList()
    torch.save(input_list[0], f"{args.output}/cali_{rar_model_size}_{number}.pth")

    globalvar.removeInput()
    torch.cuda.empty_cache()
    logger.info("down!")