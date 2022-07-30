import torch
import torch.nn as nn


class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()

        self.NNet = nn.Sequential(
            nn.Conv1d(2, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(11, stride=1),
            nn.Conv1d(32, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(11, stride=1),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(11, stride=1),
            nn.Flatten(),  # torch.Size([64,5760])
            nn.Linear(5760, 200))

        self.FNet = nn.Sequential(
            nn.Linear(4, 100),
            nn.PReLU(),
            nn.Linear(100, 100),
            nn.PReLU(),
            nn.Linear(100, 200),
            nn.PReLU())  # torch.Size([64,960])

        self.FinalNet = nn.Sequential(
            nn.Linear(400, 400),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(400, 400),
            nn.PReLU(),
            nn.Linear(400, 200),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(200, 100),
            nn.PReLU(),
            nn.Linear(100, 2),
            nn.PReLU())  # torch.Size([64,2])

    def forward(self, input):  # input = (64,248)
        Temp1 = input[:, 0:120]  # (64, 120)
        Temp2 = input[:, 120:240]  # (64, 120)
        Qualty = input[:, 240:]

        Temp3 = torch.cat([Temp1.unsqueeze(1), Temp2.unsqueeze(1)], 1)

        # print(Temp3.size())  # torch.Size([64, 2, 120])
        x = self.NNet(Temp3)

        # print('x.size:', x.size())
        q = self.FNet(Qualty)
        # print('q.size:', q.size())
        All_x = torch.cat([x, q], 1)
        # print('进入最终全连接', All_x.size())  # torch.Size([64, 1920])
        res = self.FinalNet(All_x)
        # print('result', res.size())

        return res
