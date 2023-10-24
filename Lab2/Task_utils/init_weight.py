from torch import nn

def init_weights_ones(m):
    if isinstance(m, nn.Linear):
        nn.init.ones_(m.weight.data)
        nn.init.ones_(m.bias.data)

def init_weights_W_GU_b_Z(m):
    if isinstance(m, nn.Linear):
        nn.init.glorot_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)