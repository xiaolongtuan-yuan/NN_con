import torch
import numpy as np
from model import TotalNet
import xlrd

key_list = ["x1", "x2", "x3", "x4", "x5", "x6", "y1", "y2", "y3", "y4"]

Quality = [52.75, 96.87, 46.61, 22.91]


t1 = [853.57] * 120
t2 = [766.2] * 120
q = [49.24, 90.38, 46.13, 28.16]
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
