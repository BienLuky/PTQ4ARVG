import torch
import torch.nn as nn
import numpy as np


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, leaf_param: bool = False):
        super(UniformQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.leaf_param = leaf_param
    
    def __repr__(self):
        s = super(UniformQuantizer, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={})".format(self.inited, self.channel_wise)
        return s

    def update_params(self, delta, zero_point):
        if self.delta is None:
            self.delta = delta
            self.zero_point = zero_point
        self.delta = 0.1 * delta + 0.9 * self.delta
        self.zero_point = 0.1 * zero_point + 0.9 * self.zero_point
        return self.delta, self.zero_point

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.delta, self.zero_point = self.update_params(delta, zero_point)
            # self.inited = True

        # start quantization
        x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

    def calculate_tensor(self, x: torch.Tensor, delta, zero_point):
        x_int = torch.round(x / delta) + zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - zero_point) * delta
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            best_score = 1e+10
            for pct in [0.999, 0.9999, 0.99999]:
                try:
                    new_max = torch.quantile(x_clone.reshape(-1), pct)
                    new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                except:
                    new_max = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), pct * 100),
                        device=x_clone.device,
                        dtype=torch.float32)
                    new_min = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                        device=x_clone.device,
                        dtype=torch.float32)   
                x_q = self.quantize(x_clone, new_max, new_min)
                score = lp_loss(x_clone, x_q, p=2, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / delta).round()

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q


class TokenAwareUniformQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, leaf_param: bool = False):
        super(TokenAwareUniformQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.current_token = 0
        self.token_nums = (4, 9, 16, 25, 36, 64, 100, 169, 256)
        self.leaf_param = leaf_param
    
    def __repr__(self):
        s = super(TokenAwareUniformQuantizer, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={})".format(self.inited, self.channel_wise)
        return s

    def update_params(self, delta, zero_point):
        if self.delta is None:
            self.delta = delta
            self.zero_point = zero_point
        self.delta = 0.1 * delta + 0.9 * self.delta
        self.zero_point = 0.1 * zero_point + 0.9 * self.zero_point
        return self.delta, self.zero_point

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            delta_list = []
            zero_point_list = []
            if self.channel_wise:
                for idx in range(1):
                    delta, zero_point = self.init_quantization_scale(x[:,idx,:], channel_wise=False)
                    delta_list.append(delta.view(1))
                    zero_point_list.append(zero_point.view(1))
                start_token = 1
                for token_num in self.token_nums:
                    end_token = start_token + token_num
                    delta, zero_point = self.init_quantization_scale(x[:,start_token:end_token,:], channel_wise=False)
                    delta_list.append(delta.view(1))
                    zero_point_list.append(zero_point.view(1))
                    start_token = end_token
            else:
                for idx in range(1):
                    delta, zero_point = self.init_quantization_scale(x[:,idx,:], channel_wise=False)
                    delta_list.append(delta.view(1))
                    zero_point_list.append(zero_point.view(1))
                delta, zero_point = self.init_quantization_scale(x[:,1:,:], channel_wise=False)
                delta_list.append(delta.view(1))
                zero_point_list.append(zero_point.view(1))
            delta, zero_point = torch.cat(delta_list), torch.cat(zero_point_list)
            self.delta, self.zero_point = self.update_params(delta, zero_point)
            # self.inited = True

        # start quantization
        token_num = x.size(1)
        # inference
        if token_num <= 1: # sink-token
            delta = self.delta[:token_num]
            zero_point = self.zero_point[:token_num]
            delta = delta.reshape(1, -1, 1)
            zero_point = zero_point.reshape(1, -1, 1)
        elif token_num > 1 and self.channel_wise is False: # only sink-token
            if token_num in self.token_nums: # infer
                delta = self.delta[1]
                zero_point = self.zero_point[1]
            else: # cali
                delta = torch.cat([self.delta[:1], self.delta[1].repeat(token_num-1)])
                zero_point = torch.cat([self.zero_point[:1], self.zero_point[1].repeat(token_num-1)])
            delta = delta.reshape(1, -1, 1)
            zero_point = zero_point.reshape(1, -1, 1)
        elif token_num > 1 and self.channel_wise is True: # sink-token + token-wise
            if token_num in self.token_nums: # infer
                delta = self.delta[1+self.token_nums.index(token_num)]
                zero_point = self.zero_point[1+self.token_nums.index(token_num)]
            else: # cali
                deltas, zero_points = [], []
                deltas.append(self.delta[:1])
                zero_points.append(self.zero_point[:1])
                for i in range(len(self.token_nums)):
                    delta = self.delta[1+i].repeat(self.token_nums[i])
                    zero_point = self.zero_point[1+i].repeat(self.token_nums[i])
                    deltas.append(delta)
                    zero_points.append(zero_point)
                delta = torch.cat(deltas)
                zero_point = torch.cat(zero_points)
            delta = delta.reshape(1, -1, 1)
            zero_point = zero_point.reshape(1, -1, 1)

        x_int = torch.round(x / delta) + zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - zero_point) * delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            assert len(x.shape) == 3
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[1] 
            x_max = x_clone.abs().max(dim=-1)[0].max(dim=0)[0]

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point token-by-token
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c,:], channel_wise=False)
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            best_score = 1e+10
            for pct in [0.999, 0.9999, 0.99999]:
                try:
                    new_max = torch.quantile(x_clone.reshape(-1), pct)
                    new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                except:
                    new_max = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), pct * 100),
                        device=x_clone.device,
                        dtype=torch.float32)
                    new_min = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                        device=x_clone.device,
                        dtype=torch.float32)   
                x_q = self.quantize(x_clone, new_max, new_min)
                score = lp_loss(x_clone, x_q, p=2, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / delta).round()
        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

