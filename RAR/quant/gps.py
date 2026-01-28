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

            if 'adaLN_modulation' in module_name and "adaln_before_head" not in name:
                logger.info(name)
                adaLN_module = module[1]
                next_module1 = father_module.attn.qkv
                next_module2 = father_module.mlp.fc1
                for m in [next_module1, next_module2]:
                    r, b, m.input_quantizer.delta, m.input_quantizer.zero_point = \
                        m.SmoothModule.init_sym_shift(m.input_quantizer.delta, m.input_quantizer.zero_point, m.input_quantizer.n_bits)

                    if m.bias is not None:
                        m.bias.data = m.bias.data + torch.mm(m.weight.data, b.reshape(-1,1)).reshape(-1)
                    else:
                        m.bias = Parameter(torch.Tensor(m.out_features))
                        m.bias.data = torch.mm(m.weight.data, b.reshape(-1,1)).reshape(-1)

                    m.input_quantizer.channel_wise = True
                    m.input_quantizer.inited = False
                    m.weight_quantizer.inited = False
                # absorb
                p1 = next_module1.SmoothModule.shifts
                s1 = next_module1.SmoothModule.scales
                ps1 = p1 * s1
                p2 = next_module2.SmoothModule.shifts
                s2 = next_module2.SmoothModule.scales
                ps2 = p2 * s2

                add_bias = torch.cat([-ps1, torch.full_like(ps1, 0), torch.full_like(ps1, 0), -ps2, torch.full_like(ps2, 0), torch.full_like(ps2, 0)])
                scale = torch.cat([s1, s1, torch.full_like(ps1, 1), s2, s2, torch.full_like(ps2, 1)])
                father_module.original_constant1 = father_module.original_constant1 / s1
                father_module.original_constant2 = father_module.original_constant2 / s2
                adaLN_module.weight.data = adaLN_module.weight.data / scale.reshape(-1,1)
                adaLN_module.bias.data = (adaLN_module.bias.data + add_bias) / scale

                next_module1.SmoothModule.scales = None
                next_module1.SmoothModule.shifts = None
                next_module2.SmoothModule.scales = None
                next_module2.SmoothModule.shifts = None
                adaLN_module.weight_quantizer.inited = False

            if "adaln_before_head" in module_name:
                adaLN_module = module.adaLN_modulation[1]
                m = father_module.lm_head

                r, b, m.input_quantizer.delta, m.input_quantizer.zero_point = \
                    m.SmoothModule.init_sym_shift(m.input_quantizer.delta, m.input_quantizer.zero_point, m.input_quantizer.n_bits)

                if m.bias is not None:
                    m.bias.data = m.bias.data + torch.mm(m.weight.data, b.reshape(-1,1)).reshape(-1)
                else:
                    m.bias = Parameter(torch.Tensor(m.out_features))
                    m.bias.data = torch.mm(m.weight.data, b.reshape(-1,1)).reshape(-1)

                m.input_quantizer.channel_wise = True
                m.input_quantizer.inited = False
                m.weight_quantizer.inited = False
                # absorb
                p = m.SmoothModule.shifts
                s = m.SmoothModule.scales
                ps = p*s

                add_bias = torch.cat([torch.full_like(ps, 0), -ps])
                scale = torch.cat([s, s])
                module.original_constant = module.original_constant / s
                adaLN_module.weight.data = adaLN_module.weight.data / scale.reshape(-1,1)
                adaLN_module.bias.data = (adaLN_module.bias.data + add_bias) / scale

                m.SmoothModule.scales = None
                m.SmoothModule.shifts = None
                adaLN_module.weight_quantizer.inited = False


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

            if 'adaLN_modulation' in module_name and "adaln_before_head" not in name:
                logger.info(name)
                adaLN_module = module[1]
                next_module1 = father_module.attn.qkv
                next_module2 = father_module.mlp.fc1
                for m in [next_module1, next_module2]:

                    from quant.quantizer import TokenAwareUniformQuantizer
                    m.input_quantizer = TokenAwareUniformQuantizer(**m.input_quant_params)
                    m.input_quantizer.channel_wise = False
                    m.input_quantizer.inited = False
                    m.weight_quantizer.inited = False
                    m.sink_aware = False
                # absorb
                p1 = next_module1.SmoothModule.shifts
                s1 = next_module1.SmoothModule.scales
                ps1 = p1 * s1
                p2 = next_module2.SmoothModule.shifts
                s2 = next_module2.SmoothModule.scales
                ps2 = p2 * s2

                add_bias = torch.cat([-ps1, torch.full_like(ps1, 0), torch.full_like(ps1, 0), -ps2, torch.full_like(ps2, 0), torch.full_like(ps2, 0)])
                scale = torch.cat([s1, s1, torch.full_like(ps1, 1), s2, s2, torch.full_like(ps2, 1)])
                father_module.original_constant1 = father_module.original_constant1 / s1
                father_module.original_constant2 = father_module.original_constant2 / s2
                adaLN_module.weight.data = adaLN_module.weight.data / scale.reshape(-1,1)
                adaLN_module.bias.data = (adaLN_module.bias.data + add_bias) / scale

                next_module1.SmoothModule.scales = None
                next_module1.SmoothModule.shifts = None
                next_module2.SmoothModule.scales = None
                next_module2.SmoothModule.shifts = None
                adaLN_module.weight_quantizer.inited = False

            if "adaln_before_head" in module_name:
                adaLN_module = module.adaLN_modulation[1]
                m = father_module.lm_head

                r, b, m.input_quantizer.delta, m.input_quantizer.zero_point = \
                    m.SmoothModule.init_sym_shift(m.input_quantizer.delta, m.input_quantizer.zero_point, m.input_quantizer.n_bits)
                m.weight.data = m.weight.data * r

                if m.bias is not None:
                    m.bias.data = m.bias.data + torch.mm(m.weight.data, b.reshape(-1,1)).reshape(-1)
                else:
                    m.bias = Parameter(torch.Tensor(m.out_features))
                    m.bias.data = torch.mm(m.weight.data, b.reshape(-1,1)).reshape(-1)

                m.input_quantizer.channel_wise = False
                m.input_quantizer.inited = False
                m.weight_quantizer.inited = False
                # absorb
                p = m.SmoothModule.shifts
                s = m.SmoothModule.scales
                ps = p*s

                add_bias = torch.cat([torch.full_like(ps, 0), -ps])
                scale = torch.cat([s, s])
                module.original_constant = module.original_constant / s
                adaLN_module.weight.data = adaLN_module.weight.data / scale.reshape(-1,1)
                adaLN_module.bias.data = (adaLN_module.bias.data + add_bias) / scale

                m.SmoothModule.scales = None
                m.SmoothModule.shifts = None
                adaLN_module.weight_quantizer.inited = False
