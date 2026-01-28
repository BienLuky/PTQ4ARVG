import logging
import torch

logger = logging.getLogger(__name__)

class AttentionMap:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        # self.ac_output = []

    def hook_fn(self, module, input, output):
        self.out = output
        self.feature = input
        # self.ac_output.append(output)

    def remove(self):
        self.hook.remove()


def seed_everything(seed):
    '''
    :param seed:
    :param device:
    :return:
    '''
    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
