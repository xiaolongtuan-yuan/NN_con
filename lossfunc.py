import torch
from torch import nn


class AvgStdLoss(nn.Module):
    def __init__(self):
        super(AvgStdLoss, self).__init__()

    def forward(self, output, target):
        # print('target', target.size(), 'output', output.size())
        # std = torch.div((output[:, 0] - target), torch.exp(output[:, 1]))
        std = torch.div((target - output[:, 0]), output[:, 1])
        res = torch.abs(torch.mean(std)) + torch.abs(torch.std(std) - 1)
        # res = torch.abs(torch.mean(std)) + torch.abs(torch.pow(torch.std(std), 2) - 1)
        # res = torch.abs(torch.mean(std)) + torch.abs(torch.std(std) - 1)
        return res