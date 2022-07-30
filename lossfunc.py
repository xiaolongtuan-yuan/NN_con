import torch
from torch import nn

alpha = 0


class AvgStdLoss(nn.Module):
    def __init__(self):
        super(AvgStdLoss, self).__init__()
    
    # def forward(self, output, target):
    #     # print('target', target.size(), 'output', output.size())
    #     # std = torch.div((output[:, 0] - target), torch.exp(output[:, 1]))
    #     # std = torch.div((target - output[:, 0]), torch.clamp(output[:, 1], 1e-3))
    #     std = torch.div((target - output[:, 0]), torch.abs(output[:, 1]))
    #     res = torch.mean(torch.pow(std, 2)) + torch.abs(torch.median(std)) + torch.mul(
    #         torch.pow(torch.var(std) - 1, 2), torch.mean(torch.abs(output[:, 1])))
    #     # res = torch.mean(torch.pow(std, 2))
    #     return res, std
    
    def forward(self, output, target):
        q = torch.distributions.Normal(loc=output[:, 0], scale=torch.abs(output[:, 1]))
        neg_log = -1 * torch.mean(q.log_prob(target))
        std = torch.div((target - output[:, 0]), torch.abs(output[:, 1]))

        return neg_log, std
