import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

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
                # token-aware after repq
                if self.sink_aware:
                    self.SmoothModule.init_smooth_scale(x[:,64:,:], self.weight, self.input_quantizer, self.weight_quantizer)
                else:
                    self.SmoothModule.init_smooth_scale(x, self.weight, self.input_quantizer, self.weight_quantizer)

                self.weight.data = self.weight.data * self.SmoothModule.scales
                self.SmoothModule.inited = True

            if self.SmoothModule.scales is not None:
                x = x / self.SmoothModule.scales
                x = x - self.SmoothModule.shifts

        if self.use_input_quant:
            if self.sink_aware:
                sink_x = x[:,:64,:]
                other_x = self.input_quantizer(x[:,64:,:])
                x = torch.cat((sink_x, other_x), dim=-2)
            else:
                x = self.input_quantizer(x)

        if self.use_weight_quant:
            w = self.weight_quantizer(self.weight)
        else:
            w = self.weight

        out = F.linear(x, weight=w, bias=self.bias)

        return out
        