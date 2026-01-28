import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import UniformQuantizer, TokenAwareUniformQuantizer
from .smooth import SmoothModule


class QuantLinear(nn.Linear):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 in_features,
                 out_features,
                 input_quant_params={},
                 weight_quant_params={},
                 token_aware=False,
                 smooth=False):
        super(QuantLinear, self).__init__(in_features, out_features)
        self.input_quant_params = input_quant_params
        if token_aware:
            self.input_quantizer = TokenAwareUniformQuantizer(**input_quant_params)
        else:
            self.input_quantizer = UniformQuantizer(**input_quant_params)
        self.weight_quantizer = UniformQuantizer(**weight_quant_params)

        if smooth:
            self.SmoothModule = SmoothModule()

        self.use_input_quant = False
        self.use_weight_quant = False
        self.sink_aware = False
        self.ori_weight = None

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s
    
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def forward(self, x):
        """
        using quantized weights to forward input x
        """
        if hasattr(self, 'SmoothModule'):
            if self.SmoothModule.inited == False:
                if self.ori_weight == None:
                    self.ori_weight = self.weight.data.clone()

                if self.sink_aware:
                    self.SmoothModule.init_smooth_scale(x[:,1:,:], self.ori_weight, self.input_quantizer, self.weight_quantizer)
                else:
                    self.SmoothModule.init_smooth_scale(x, self.ori_weight, self.input_quantizer, self.weight_quantizer)

                self.weight.data = self.ori_weight * self.SmoothModule.scales

            if self.SmoothModule.scales is not None:
                x = x / self.SmoothModule.scales
                x = x - self.SmoothModule.shifts

        if self.use_input_quant:
            if self.sink_aware:
                sink_x = x[:,:1,:]
                other_x = self.input_quantizer(x[:,1:,:])
                x = torch.cat((sink_x, other_x), dim=-2)
            else:
                x = self.input_quantizer(x)

        if self.use_weight_quant:
            w = self.weight_quantizer(self.weight)
        else:
            w = self.weight

        out = F.linear(x, weight=w, bias=self.bias)

        return out
        

class QuantMatMul(nn.Module):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 input_quant_params={}):
        super(QuantMatMul, self).__init__()

        self.quantizer_A = UniformQuantizer(**input_quant_params)
        self.quantizer_B = UniformQuantizer(**input_quant_params)

        self.use_input_quant = False

    def __repr__(self):
        s = super(QuantMatMul, self).__repr__()
        s = "(" + s + "input_quant={})".format(self.use_input_quant)
        return s
    
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant

    def forward(self, A, B):
        if self.use_input_quant:
            A = self.quantizer_A(A)
            B = self.quantizer_B(B)
        
        out = A @ B
        return out
