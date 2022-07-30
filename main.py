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
    learning_rate = 3e-5
    device = torch.device("cuda")

    model = TotalNet()  # 模型
    model = model.to(device)
    loss_fn = AvgStdLoss()
    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10)

    total_dataset = Folder(index=y)
    train_size = int(0.75 * len(total_dataset))
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

    writer = SummaryWriter("logs_train_y" + str(y))

    for i in range(epoch):
        print("------第{}轮训练开始------".format(i + 1))

        model.train()
        for j, (inputs, targets) in enumerate(train_loader):

            inputs = inputs.to(torch.float32)
            inputs = inputs.to(device)
            targets = targets.to(torch.float32)
            targets = targets.to(device)

            # print("input2": input, "target2": targets)

            outputs = model(inputs)
            loss, stds = loss_fn(outputs, targets)

            #  优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
            optimizer.step()
            total_train_step += 1
            if total_train_step % 20 == 0:
                print("训练次数：{}, Loss:{}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)
                writer.add_histogram("train_Output_stds", stds, total_train_step)

        model.eval()
        with torch.no_grad():

            total_test_loss = 0
            num = 0
            for j, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(torch.float32)
                inputs = inputs.to(device)
                targets = targets.to(torch.float32)
                targets = targets.to(device)
                outputs = model(inputs)

                loss, stds = loss_fn(outputs, targets)
                num += 1
                total_test_loss += loss.item()

            avg_test_loss = total_test_loss / num
            scheduler.step(avg_test_loss)

            print("测试集Loss：{}".format(avg_test_loss))
            writer.add_scalar("test_loss", avg_test_loss, i)
            writer.add_histogram("test_Output_stds", stds, total_train_step)

    torch.save(model, "models/longtuan_y" + str(y) + ".pth")
    print("模型已保存！")

    writer.close()
