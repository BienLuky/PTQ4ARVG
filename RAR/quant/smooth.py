import torch
import torch.nn as nn

class SmoothModule(nn.Module):
    def __init__(self):
        super(SmoothModule, self).__init__()
        self.scales = None
        self.shifts = None
        self.inited = True
        self.calculate_optimal_factor = False

    def calculate_scales(self, input, weight, alpha):
        if input.dim() == 3:
            tensor_permuted = input.permute(2, 0, 1)
        else:
            tensor_permuted = input.permute(1, 0)
        l = tensor_permuted.reshape(tensor_permuted.size(0), -1)
        max_input = l.abs().max(dim=1)[0].clamp(min=1e-6)
        max_weight = weight.abs().max(dim=0)[0].clamp(min=1e-6)

        scales = max_input.pow(alpha) / max_weight.pow(1 - alpha)
        shifts = torch.full_like(scales, 0)
        return scales, shifts

    def calculate_scales_hessian(self, input, weight, alpha, input_quantizer, weight_quantizer):
        # first smooth get s1 based on range of act
        if input.dim() == 3:
            tensor_permuted = input.permute(2, 0, 1)
        else:
            tensor_permuted = input.permute(1, 0)
        l = tensor_permuted.reshape(tensor_permuted.size(0), -1) # l = Ci * (N*T)

        max_input, min_input = l.max(dim=1)[0], l.min(dim=1)[0]
        range_input = max_input - min_input
        max_weight, min_weight = weight.max(dim=0)[0], weight.min(dim=0)[0]
        range_weight = max_weight - min_weight
        max_input_range, idx = torch.max(range_input, dim=0)

        scale_std = max_input_range.pow(alpha) / range_weight[idx].pow(1 - alpha)

        # secend calculate factor
        scales = torch.full_like(max_input, 1)
        shifts = torch.full_like(scales, 0)

        act_delta = input_quantizer.delta.reshape(-1)
        act_zero_point = input_quantizer.zero_point.reshape(-1)
        q_l = input_quantizer.calculate_tensor(l, act_delta[idx], act_zero_point[idx]) # q_input = Ci * (N*T)
        q_weight = weight_quantizer(weight) # q_weight = Co * Ci
        err_l = torch.abs(l - q_l)
        err_weight = torch.abs(weight - q_weight)

        for i in range(len(scales)):
            if i != idx:
                frac_11 = torch.sum(torch.square(l[i]))
                frac_12 = torch.sum(torch.square(err_weight[:,i]))
                frac_1 = frac_11 * frac_12
                frac_21 = torch.sum(torch.square(err_l[i]))
                frac_22 = torch.sum(torch.square(q_weight[:,i])) # Strict should q_weight instead of weight
                frac_2 = frac_21 * frac_22
                if frac_2 == 0: # Avoid dividing negative value. frac_2 == 0 means that the return is always negative, so maximizing scale minimizes losses
                    scales[i] = torch.tensor(1.0)
                else:
                    s = (scale_std ** 4) * (frac_1 / frac_2)
                    scales[i] = torch.sqrt(torch.sqrt(s))

            else:
                scales[i] = scale_std

        return scales, shifts

    def calculate_sym_shift(self, input_s, input_zp, bit_width):
        input_s = input_s.reshape(-1)
        input_zp = input_zp.reshape(-1)
        act_min = -input_zp * input_s
        
        target_s = input_s
        target_zp = 2 ** (bit_width-1) - 1
        target_min = -target_zp * target_s

        scales = input_s / target_s
        shifts = act_min / scales - target_min
        return scales, shifts, target_s, target_zp

    def init_sym_shift(self, input_s: torch.Tensor, input_zp: torch.Tensor, bit_width: int):
        '''sym_shift'''
        scales, shifts, target_s, target_zp  = self.calculate_sym_shift(input_s, input_zp, bit_width)
        if self.scales is None:
            self.scales = scales
            self.shifts = shifts
        else:
            self.scales = self.scales * scales
            self.shifts = self.shifts + shifts
        return scales, shifts, target_s, target_zp

    def init_smooth_scale(self, input, weight, input_quantizer=None, weight_quantizer=None):
        if self.calculate_optimal_factor:
            '''GPS'''
            self.scales, self.shifts = self.calculate_scales_hessian(input, weight, alpha=0.5, input_quantizer=input_quantizer, weight_quantizer=weight_quantizer)
        else:
            '''smoothquant'''
            self.scales, self.shifts = self.calculate_scales(input, weight, alpha=0.5)
        return self.scales, self.shifts

