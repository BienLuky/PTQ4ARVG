import logging
import torch

from quant.quantizer import UniformQuantizer, TokenAwareUniformQuantizer, LogSqrt2Quantizer
from quant.quant_modules import QuantLinear
from quant.quant_model import set_quant_state
logger = logging.getLogger(__name__)


def set_smooth_quantize_params(q_model, hs, freqs_cis, input_pos, mask, batch_size: int = 256,):
    logger.info(f"set_smooth_quantize_params")

    batch_size = min(batch_size, hs.size(0))
    set_quant_state(q_model, input_quant=False, weight_quant=False)
    for name, module in q_model.named_modules():
        if isinstance(module, QuantLinear) and hasattr(module, 'SmoothModule'):
            module.SmoothModule.inited = False

    with torch.no_grad():
        for i in range(int(hs.size(0) / batch_size)):
            q_model.forward_fn(hs[i*batch_size : (i+1)*batch_size], freqs_cis, input_pos, mask[i*batch_size : (i+1)*batch_size])
    torch.cuda.empty_cache()

    for name, module in q_model.named_modules():
        if isinstance(module, QuantLinear) and hasattr(module, 'SmoothModule'):
            module.SmoothModule.inited = True
            module.weight.data = module.ori_weight.data * module.SmoothModule.scales
            module.ori_weight = None


def set_calculating_optimal_factor(q_model, hs, freqs_cis, input_pos, mask, batch_size: int = 256,):
    logger.info(f"set_calculating_optimal_factor")

    batch_size = min(batch_size, hs.size(0))
    set_quant_state(q_model, input_quant=False, weight_quant=False)
    for name, module in q_model.named_modules():
        if isinstance(module, QuantLinear) and hasattr(module, 'SmoothModule'):
            module.SmoothModule.inited = False
            module.SmoothModule.calculate_optimal_factor = True
    
    with torch.no_grad():
        for i in range(int(hs.size(0) / batch_size)):
            q_model.forward_fn(hs[i*batch_size : (i+1)*batch_size], freqs_cis, input_pos, mask[i*batch_size : (i+1)*batch_size])
    torch.cuda.empty_cache()

    for name, module in q_model.named_modules():
        if isinstance(module, QuantLinear) and hasattr(module, 'SmoothModule'):
            module.SmoothModule.inited = True
            module.SmoothModule.calculate_optimal_factor = False
            module.weight.data = module.ori_weight.data * module.SmoothModule.scales
            module.ori_weight = None


def set_act_quantize_params(q_model, hs, freqs_cis, input_pos, mask, batch_size: int = 256,):
    logger.info(f"set_act_quantize_params")

    batch_size = min(batch_size, hs.size(0))
    set_quant_state(q_model, input_quant=True, weight_quant=True)
    for name, module in q_model.named_modules():
        if isinstance(module, (UniformQuantizer, TokenAwareUniformQuantizer, LogSqrt2Quantizer)):
            if module.leaf_param == True:
                module.inited = False
                module.delta = None
    
    with torch.no_grad():
        for i in range(int(hs.size(0) / batch_size)):
            q_model.forward_fn(hs[i*batch_size : (i+1)*batch_size], freqs_cis, input_pos, mask[i*batch_size : (i+1)*batch_size])
    torch.cuda.empty_cache()

    for name, module in q_model.named_modules():
        if isinstance(module, (UniformQuantizer, TokenAwareUniformQuantizer, LogSqrt2Quantizer)):
            if module.leaf_param == True:
                module.inited = True


def set_weight_quantize_params(q_model, hs, freqs_cis, input_pos, mask):
    logger.info(f"set_weight_quantize_params")

    set_quant_state(q_model, input_quant=False, weight_quant=True)
    for name, module in q_model.named_modules():
        if isinstance(module, UniformQuantizer):
            if module.leaf_param == False:
                module.inited = False
                module.delta = None

    with torch.no_grad():
        q_model.forward_fn(hs[:2], freqs_cis, input_pos, mask[:2])
    torch.cuda.empty_cache()

    for name, module in q_model.named_modules():
        if isinstance(module, UniformQuantizer):
            if module.leaf_param == False:
                module.inited = True


def reset_act_quantize_params(q_model, hs, freqs_cis, input_pos, mask, batch_size: int = 256,):
    logger.info(f"reset_act_quantize_params")

    batch_size = min(batch_size, hs.size(0))
    set_quant_state(q_model, input_quant=True, weight_quant=True)
    
    with torch.no_grad():
        for i in range(int(hs.size(0) / batch_size)):
            q_model.forward_fn(hs[i*batch_size : (i+1)*batch_size], freqs_cis, input_pos, mask[i*batch_size : (i+1)*batch_size])
    torch.cuda.empty_cache()

    for name, module in q_model.named_modules():
        if isinstance(module, (UniformQuantizer, TokenAwareUniformQuantizer, LogSqrt2Quantizer)):
            if module.leaf_param == True:
                module.inited = True


def reset_weight_quantize_params(q_model, hs, freqs_cis, input_pos, mask):
    logger.info(f"reset_weight_quantize_params")

    set_quant_state(q_model, input_quant=False, weight_quant=True)

    with torch.no_grad():
        q_model.forward_fn(hs[:2], freqs_cis, input_pos, mask[:2])
    torch.cuda.empty_cache()

    for name, module in q_model.named_modules():
        if isinstance(module, UniformQuantizer):
            if module.leaf_param == False:
                module.inited = True

