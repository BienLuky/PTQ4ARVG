import logging
import torch

from quant.quant_modules import QuantLinear
from quant.quant_model import set_quant_state
logger = logging.getLogger(__name__)


def set_smooth_quantize_params(q_model, tokens, mask, class_embedding):
    logger.info(f"set_smooth_quantize_params")

    set_quant_state(q_model, input_quant=False, weight_quant=False)
    for name, module in q_model.named_modules():
        if isinstance(module, QuantLinear) and hasattr(module, 'SmoothModule'):
            module.SmoothModule.inited = False

    with torch.no_grad():
        _ = q_model.forward_fn(tokens, mask, class_embedding)
    torch.cuda.empty_cache()

    for name, module in q_model.named_modules():
        if isinstance(module, QuantLinear) and hasattr(module, 'SmoothModule'):
            module.SmoothModule.inited = True


def set_calculating_optimal_factor(q_model, tokens, mask, class_embedding):
    logger.info(f"set_calculating_optimal_factor")

    set_quant_state(q_model, input_quant=False, weight_quant=False)
    for name, module in q_model.named_modules():
        if isinstance(module, QuantLinear) and hasattr(module, 'SmoothModule'):
            module.SmoothModule.inited = False
            module.SmoothModule.calculate_optimal_factor = True
    
    with torch.no_grad():
        _ = q_model.forward_fn(tokens, mask, class_embedding)
    torch.cuda.empty_cache()

    for name, module in q_model.named_modules():
        if isinstance(module, QuantLinear) and hasattr(module, 'SmoothModule'):
            module.SmoothModule.inited = True
            module.SmoothModule.calculate_optimal_factor = False

