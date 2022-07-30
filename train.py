from folders import Folder
from model import TotalNet
from torch.utils import data
from lossfunc import AvgStdLoss
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

if __name__ == '__main__':
    y = 4  # 预测指标
    epoch = 500
    batch_size = 512
    learning_rate = 3e-3
    device = torch.device("cuda")

    model = TotalNet()  # 模型
    model = model.to(device)
    loss_fn = AvgStdLoss()
    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10)

    total_dataset = Folder(index=y)
    train_size = int(0.7 * len(total_dataset))
    valid_size = len(total_dataset) - train_size
    train_dataset, test_dataset = data.random_split(
        total_dataset, [train_size, valid_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)
    # 模型初始化
    # model.apply(weights_init_xavier)

    train_len = len(train_dataset)
    test_len = len(test_dataset)
    print(train_len)
    print(test_len)

    total_train_step = 0

    model = torch.load('models/longtuan_y1.pth').to(device)
    for j, (inputs, targets) in enumerate(train_loader):

        model.eval()
        with torch.no_grad():
            inputs = inputs.to(torch.float32)
            inputs = inputs.to(device)
            targets = targets.to(torch.float32)
            targets = targets.to(device)

            # print("input2": input, "target2": targets)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            print('outputA: ', outputs)
            print('loss', loss)

