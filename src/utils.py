# import tensorflow as tf
import torch
import torch.nn as nn

from logHandler import get_logger

def print_netowrk(net):
    logger = get_logger(__name__)
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    logger.info(net)
    logger.info('Total number of parameters: {}'.format(num_params))

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.2)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.2)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.2)
            m.bias.data.zero_()
