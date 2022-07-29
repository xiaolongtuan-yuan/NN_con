import torch
import numpy as np
from model import TotalNet

key_list = ["x1", "x2", "x3", "x4", "x5", "x6", "y1", "y2", "y3", "y4"]
max_values = [1410.47, 987.42, 55.11, 119.4, 49.03, 28.16, 82.79, 28.0, 13.97, 27.38]
min_values = [296.16, 518.08, 47.95, 86.09, 41.97, 19.69, 77.36, 20.08, 9.82, 11.77]
max_values = np.asarray(max_values)
min_values = np.asarray(min_values)


def data_scale(input):
    # 归一化
    for j in range(len(input)):
        if j < 120:
            input[j] = (input[j] - min_values[0]) / (max_values[0] - min_values[0])
        elif j < 240:
            input[j] = (input[j] - min_values[1]) / (max_values[1] - min_values[1])
        else:
            input[j] = (input[j] - min_values[j - 238]) / (max_values[j - 238] - min_values[j - 238])
    return input

def anti_data_scale(input, i):
    # 特征缩放还原,还原4个数据
    # for i in range(4):
    #     input[i] = input[i] * (max_values[i + 6] - min_values[i + 6]) + min_values[i + 6]
    # return input
    # 返回[y1,y2,y3,y4]
    res = input * (max_values[6+i] - min_values[6+i]) + min_values[6+i]
    return res


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

    output = anti_data_scale(output[0], i-1)

    print("y"+str(i)+":  ", output[0])

