import torch
import numpy as np
import csv
from model import TotalNet

batch_size = 512
q = [52.75, 96.87, 46.61, 22.91]
b = np.loadtxt("data/test.csv", delimiter=",")
device = torch.device("cuda")
shape = b.shape
index = 0
data = []
for i in range(shape[0]):
    t1 = [b[i, 0]] * 120
    t2 = [b[i, 1]] * 120
    r = t1 + t2 + q
    data.append(r)
print(len(data))
batchs = []
batch = []
for i in range(len(data)):
    if (i % batch_size == 0 and i != 0):
        batchs.append(batch)
        batch = []
        batch.append(data[i])
    else:
        batch.append(data[i])
batchs.append(batch)
# print(batchs)
forecast = []  # 所有输出
models = []
for i in range(1, 5):
    model = torch.load('./models/longtuan_y' + str(i) + '.pth').to(device)
    models.append(model)

for input in batchs:
    input = np.asarray(input)
    input = torch.from_numpy(input).to(torch.float32).to(device)
    outputs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            print('input', input)
            output = model(input)
            print('output', output)
        outputs.append(output)
    forecast.append(outputs)


if 1 == 0:
    data = t1 + t2 + q
    # [855.03, 768.16, 49.24, 90.38, 46.13, 28.16]

    data = np.asarray(data)

    data = data_scale(data)

    data = torch.from_numpy(data).type(torch.FloatTensor).unsqueeze(0)

    data = data.cuda()

    print("data", data)

    for i in range(1, 5):
        model = torch.load('./models/longtuan_y' + str(i) + '.pth').cuda()
        model.eval()
        with torch.no_grad():
            output = model(data)
        print("y" + str(i) + ":  ", output[0])

        output = anti_data_scale(output[0], i - 1)

        print("y" + str(i) + ":  ", output[0])
