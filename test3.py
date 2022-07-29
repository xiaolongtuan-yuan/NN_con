from dataloader import DataLoader
from model import TotalNet
from lossfunc import AvgStdLoss

from torch.nn import functional as F
import torch.optim
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    train_loader = DataLoader(64, istrain=True)
    test_loader = DataLoader(64, istrain=False)
    train_data = train_loader.get_data()
    test_data = test_loader.get_data()

    # 模型初始化
    model = TotalNet().cuda()
    # model.apply(weights_init_xavier)

    train_len = len(train_loader.data)
    test_len = len(test_loader.data)


    longtuan = TotalNet()  # 模型
    longtuan = longtuan.cuda()

    for j, (inputs, targets) in enumerate(train_data):

        inputs = inputs.to(torch.float32)
        inputs = inputs.cuda()
        targets = targets.to(torch.float32)
        targets = targets.cuda()

        print("input2:", inputs.size(), "target2:", targets.size())

        outputs = longtuan(inputs)
        break