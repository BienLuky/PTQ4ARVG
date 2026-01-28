import torch.nn as nn
from .quant_modules import QuantLinear
from copy import deepcopy


def quant_model(model, input_quant_params={}, weight_quant_params={}):

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
        
        if isinstance(m, nn.Linear) and 'diffloss' not in name:
            # Linear Layer
            idx = idx + 1 if idx != 0 else idx
            if 'qkv' in name or 'fc1' in name:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params_channel, weight_quant_params, smooth=True)
                new_m.sink_aware = True

            elif "decoder_embed" in name:
                continue
            
            elif ('proj' in name and 'z_proj' not in name)  or 'fc2' in name:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params, weight_quant_params, token_aware=True)

            else:
                continue
            
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)

    return model


def set_quant_state(model, input_quant=False, weight_quant=False):
    for m in model.modules():
        if isinstance(m, (QuantLinear)):
            m.set_quant_state(input_quant, weight_quant)
