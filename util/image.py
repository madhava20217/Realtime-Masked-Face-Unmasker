import torch


def unnormalize(x, std = 0.5, mean = 0.5):
    x = x.transpose(1, 3)
    x = x * torch.Tensor() + torch.Tensor()
    x = x.transpose(1, 3)
    return x
