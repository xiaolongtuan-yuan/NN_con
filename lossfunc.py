import torch
from torch import nn


alpha = 0.1


class AvgStdLoss(nn.Module):
    def __init__(self):
        super(AvgStdLoss, self).__init__()
    
    def forward(self, output, target):
        # print('target', target.size(), 'output', output.size())
        # std = torch.div((output[:, 0] - target), torch.exp(output[:, 1]))
        std = torch.div((target - output[:, 0]), output[:, 1])
        # res = alpha * torch.pow(torch.mean(std), 2) + (1-alpha) * torch.pow(torch.std(std) - 1, 2)
        res = torch.pow(torch.mean(std), 2) + 10 * torch.pow(torch.std(std) - 1, 2)
        return res
