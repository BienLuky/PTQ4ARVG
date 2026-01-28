import torch.nn as nn
from autoregressive.models.gpt import MatMul
from .quant_modules import QuantLinear, QuantMatMul
from copy import deepcopy


def quant_model(model, input_quant_params={}, weight_quant_params={}):
    # post-softmax
    input_quant_params_matmul2 = deepcopy(input_quant_params)
    input_quant_params_matmul2['log_quant'] = True

    # SimQuant
    input_quant_params_channel = deepcopy(input_quant_params)
    input_quant_params_channel['channel_wise'] = True

    module_dict={}
    for name, m in model.named_modules():
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        
        if isinstance(m, nn.Linear):
            # Linear Layer
            idx = idx + 1 if idx != 0 else idx
            if 'output' in name:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params_channel, weight_quant_params, smooth=True)

            elif 'wqkv' in name or 'w1' in name:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params_channel, weight_quant_params, smooth=True)
                new_m.sink_aware = True

            elif 'w3' in name:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params_channel, weight_quant_params, smooth=False)
                new_m.sink_aware = True

            elif 'wo' in name or 'w2' in name:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params, weight_quant_params, token_aware=True)

            else:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params, weight_quant_params)

            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
            
        elif isinstance(m, MatMul):
            # Matmul Layer
            idx = idx + 1 if idx != 0 else idx
            if 'matmul2' in name:
                new_m = QuantMatMul(input_quant_params_matmul2)
            else:
                new_m = QuantMatMul(input_quant_params)
            setattr(father_module, name[idx:], new_m)

    return model


def set_quant_state(model, input_quant=False, weight_quant=False):
    for m in model.modules():
        if isinstance(m, (QuantLinear, QuantMatMul)):
            m.set_quant_state(input_quant, weight_quant)
