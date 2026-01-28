'''
First, remember to uncomment line 170-171 in ./models/var.py and comment them after finish collecting.
'''
################## 1. Download checkpoints and build models
import os, logging, argparse, datetime
import torch, torchvision
from models import VQVAE, build_vae_var
from quant.utils import seed_everything
from models.basic_var import SelfAttention
from torch.nn import Parameter
logger = logging.getLogger(__name__)

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
    logdir = os.path.join(args.output, "log", now+'-VAR_cali')
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

    ############################# 2. Sample with classifier-free guidance

    # set args
    # num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
    cfg = 1.5 #@param {type:"slider", min:1, max:10, step:0.1}
    top_p = 0.96
    top_k = 900
    more_smooth = False # True for more smooth output

    # seed
    seed_everything(args.seed)

    # sample
    number = args.cali_num_samples
    batch = args.cali_batch
    input_list = []
    for i in range(int(number/batch)):
        class_labels = list(torch.randint(0, 999, size=(batch,))) # random IN-1k class
        B = len(class_labels)
        label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float32, cache_enabled=True):    # using bfloat16 can be faster
                recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=args.seed, more_smooth=more_smooth)

        '''
        First, remember to uncomment line 170-171 in ./models/var.py and comment them after finish collecting.
        '''
        import quant.globalvar as globalvar   
        input_list.append(globalvar.getInputList().copy())
        globalvar.removeInput()
        torch.cuda.empty_cache()
    
    inputs = []
    for i in range(len(input_list[0])):
        input = torch.cat([inp[i][0] for inp in input_list])
        inputs.append(input)
    conds = torch.cat([inp[0][1] for inp in input_list])
    torch.save((inputs, conds), f"{args.output}/cali_{var_model_size}_{number}.pth")
    logger.info(f"cali path: {args.output}/cali_{var_model_size}_{number}.pth")
    
    logger.info("down!")