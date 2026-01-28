import torch
from torch.nn import Parameter
import logging
logger = logging.getLogger(__name__)


def shift_activation_to_sym(q_model):
    """
    Shifting the activation channel to 0 symmetry eliminates the effect of uneven ||X||^2 distribution on scaling gains
    """
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
            
            if 'attention_norm' in module_name:
                # logger.info(name)
                adaLN_module = module
                next_module = father_module.attention.wqkv
                m = next_module
                r, b, m.input_quantizer.delta, m.input_quantizer.zero_point = \
                    m.SmoothModule.init_sym_shift(m.input_quantizer.delta, m.input_quantizer.zero_point, m.input_quantizer.n_bits)
                m.weight.data = m.weight.data * r

                if m.bias is not None:
                    m.bias.data = m.bias.data + torch.mm(m.weight.data, b.reshape(-1,1)).reshape(-1)
                else:
                    m.bias = Parameter(torch.Tensor(m.out_features))
                    m.bias.data = torch.mm(m.weight.data, b.reshape(-1,1)).reshape(-1)
                
                m.input_quantizer.channel_wise = True
                m.input_quantizer.inited = False
                m.input_quantizer.delta = None
                
                # absorb
                p = m.SmoothModule.shifts
                s = m.SmoothModule.scales
                ps = p*s

                add_bias = torch.cat([-ps])
                scale = torch.cat([s])
                adaLN_module.weight.data = adaLN_module.weight.data / scale
                adaLN_module.bias = (adaLN_module.bias + add_bias) / scale

                m.SmoothModule.scales = None
                m.SmoothModule.shifts = None

            elif 'ffn_norm' in module_name:
                # logger.info(name)
                adaLN_module = module
                next_module1 = father_module.feed_forward.w1
                next_module2 = father_module.feed_forward.w3
                m = next_module1
                r, b, m.input_quantizer.delta, m.input_quantizer.zero_point = \
                    m.SmoothModule.init_sym_shift(m.input_quantizer.delta, m.input_quantizer.zero_point, m.input_quantizer.n_bits)
                m.weight.data = m.weight.data * r

                if m.bias is not None:
                    m.bias.data = m.bias.data + torch.mm(m.weight.data, b.reshape(-1,1)).reshape(-1)
                else:
                    m.bias = Parameter(torch.Tensor(m.out_features))
                    m.bias.data = torch.mm(m.weight.data, b.reshape(-1,1)).reshape(-1)

                next_module2.weight.data = next_module2.weight.data * m.SmoothModule.scales
                if next_module2.bias is not None:
                    next_module2.bias.data = next_module2.bias.data + torch.mm(next_module2.weight.data, b.reshape(-1,1)).reshape(-1)
                else:
                    next_module2.bias = Parameter(torch.Tensor(next_module2.out_features))
                    next_module2.bias.data = torch.mm(next_module2.weight.data, b.reshape(-1,1)).reshape(-1)

                m.input_quantizer.channel_wise = True
                m.input_quantizer.inited = False
                m.input_quantizer.delta = None
                next_module2.input_quantizer.channel_wise = True
                next_module2.input_quantizer.inited = False
                next_module2.input_quantizer.delta = None
                next_module2.weight_quantizer.inited = False
                next_module2.weight_quantizer.delta = None

                # absorb
                p = m.SmoothModule.shifts
                s = m.SmoothModule.scales
                ps = p*s

                add_bias = torch.cat([-ps])
                scale = torch.cat([s])
                adaLN_module.weight.data = adaLN_module.weight.data / scale
                adaLN_module.bias = (adaLN_module.bias + add_bias) / scale

                m.SmoothModule.scales = None
                m.SmoothModule.shifts = None

            if "output" in module_name:
                # logger.info(name)
                adaLN_module = father_module.norm
                m = module
                r, b, m.input_quantizer.delta, m.input_quantizer.zero_point = \
                    m.SmoothModule.init_sym_shift(m.input_quantizer.delta, m.input_quantizer.zero_point, m.input_quantizer.n_bits)
                m.weight.data = m.weight.data * r

                if m.bias is not None:
                    m.bias.data = m.bias.data + torch.mm(m.weight.data, b.reshape(-1,1)).reshape(-1)
                else:
                    m.bias = Parameter(torch.Tensor(m.out_features))
                    m.bias.data = torch.mm(m.weight.data, b.reshape(-1,1)).reshape(-1)

                m.input_quantizer.channel_wise = True
                m.input_quantizer.inited = False
                m.input_quantizer.delta = None

                # absorb
                p = m.SmoothModule.shifts
                s = m.SmoothModule.scales
                ps = p*s

                add_bias = torch.cat([-ps])
                scale = torch.cat([s])
                adaLN_module.weight.data = adaLN_module.weight.data / scale
                adaLN_module.bias = (adaLN_module.bias + add_bias) / scale

                m.SmoothModule.scales = None
                m.SmoothModule.shifts = None


def absorb_gps_scaling_factor(q_model):
    """
    Absorb scaling factor
    """
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

            if 'attention_norm' in module_name:
                logger.info(name)
                adaLN_module = module
                next_module = father_module.attention.wqkv
                m = next_module

                from quant.quantizer import TokenAwareUniformQuantizer
                m.input_quantizer = TokenAwareUniformQuantizer(**m.input_quant_params)
                m.input_quantizer.channel_wise = False
                m.input_quantizer.inited = False
                m.input_quantizer.delta = None
                m.weight_quantizer.inited = False
                m.weight_quantizer.delta = None
                m.sink_aware = False
                
                # absorb
                p = m.SmoothModule.shifts
                s = m.SmoothModule.scales
                ps = p*s

                add_bias = torch.cat([-ps])
                scale = torch.cat([s])
                adaLN_module.weight.data = adaLN_module.weight.data / scale
                adaLN_module.bias = (adaLN_module.bias + add_bias) / scale

                m.SmoothModule.scales = None
                m.SmoothModule.shifts = None

            if 'ffn_norm' in module_name:
                logger.info(name)
                adaLN_module = module
                next_module1 = father_module.feed_forward.w1
                next_module2 = father_module.feed_forward.w3
                m = next_module1

                next_module2.weight.data = next_module2.weight.data * m.SmoothModule.scales

                from quant.quantizer import TokenAwareUniformQuantizer
                m.input_quantizer = TokenAwareUniformQuantizer(**m.input_quant_params)
                m.input_quantizer.channel_wise = False
                m.input_quantizer.inited = False
                m.input_quantizer.delta = None
                m.weight_quantizer.inited = False
                m.weight_quantizer.delta = None
                m.sink_aware = False
                next_module2.input_quantizer = TokenAwareUniformQuantizer(**m.input_quant_params)
                next_module2.input_quantizer.channel_wise = False
                next_module2.input_quantizer.inited = False
                next_module2.input_quantizer.delta = None
                next_module2.weight_quantizer.inited = False
                next_module2.weight_quantizer.delta = None
                next_module2.sink_aware = False

                # absorb
                p = m.SmoothModule.shifts
                s = m.SmoothModule.scales
                ps = p*s

                add_bias = torch.cat([-ps])
                scale = torch.cat([s])
                adaLN_module.weight.data = adaLN_module.weight.data / scale
                adaLN_module.bias = (adaLN_module.bias + add_bias) / scale

                m.SmoothModule.scales = None
                m.SmoothModule.shifts = None

            if "output" in module_name:
                logger.info(name)
                adaLN_module = father_module.norm
                m = module

                m.input_quantizer.channel_wise = False
                m.input_quantizer.inited = False
                m.input_quantizer.delta = None
                m.weight_quantizer.inited = False
                m.weight_quantizer.delta = None
                # absorb
                p = m.SmoothModule.shifts
                s = m.SmoothModule.scales
                ps = p*s

                add_bias = torch.cat([-ps])
                scale = torch.cat([s])
                adaLN_module.weight.data = adaLN_module.weight.data / scale
                adaLN_module.bias = (adaLN_module.bias + add_bias) / scale

                m.SmoothModule.scales = None
                m.SmoothModule.shifts = None
