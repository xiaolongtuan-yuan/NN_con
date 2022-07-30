import xlrd
import torch.utils.data as data
import torch
# from sklearn.preprocessing import StandardScaler

import numpy as np


# 归一化参数再record.txt

class Folder(data.Dataset):
    def __init__(self, index):
        self.index = index  # 用于标记要预测的值
        wb = xlrd.open_workbook('data/data3.xls')
        sh = wb.sheets()[0]
        
        self.data = []
        self.nrows = sh.nrows
        self.ncols = sh.ncols
        
        # self.getMax_Min(sh)
        T1_obj = sh.col(0)
        self.T1_val = []
        for cell in T1_obj:
            self.T1_val.append(cell.value)
        
        T2_obj = sh.col(1)
        self.T2_val = []
        for cell in T2_obj:
            self.T2_val.append(cell.value)
        
        # self.sh = []
        # for i in range(1, self.nrows):
        #     x = []
        #     for j in range(self.ncols):
        #         x.append(sh.cell(i,j).value)
        #     self.sh.append(x)
        
        for i in range(121, self.nrows):
            x = []
            x += self.T1_val[i - 120:i]
            x += self.T2_val[i - 120:i]
            for j in range(2, self.ncols):
                x.append(sh.cell(i, j).value)
            
            self.data.append(x)
        # (n, 248)shape
        # print(len(self.data[1]))
        self.length = len(self.data)
    
    # self.data = self.data_scale()  # 归一化
    
    def __getitem__(self, item):
        i = -(5 - self.index)
        input = torch.from_numpy(np.asarray(self.data[item][:-4]))
        if self.index != 4:
            label = torch.from_numpy(np.asarray(self.data[item][i:i + 1]))  # 值预测y
        else:
            label = torch.from_numpy(np.asarray(self.data[item][i:]))  # 值预测y
        return input, label
    
    # return self.input[item], self.label[item]  # 返回两个张量
    
    def __len__(self):
        return self.length

# def getMax_Min(self, sh):
#     self.max = [0] * 10
#     self.min = [10000] * 10
#     for i in range(1, self.nrows):
#         for j in range(self.ncols):
#             if sh.cell(i, j).value > self.max[j]:
#                 self.max[j] = sh.cell(i, j).value
#             if sh.cell(i, j).value < self.min[j]:
#                 self.min[j] = sh.cell(i, j).value
#     return 0

# def getMean_Std(self):
#     data = np.asarray(self.sh)
#     print(data.shape)
#     data = data.transpose()
#     print(data.shape)
#     means = []
#     stds = []
#     for i in range(10):
#         x = np.mean(data[i])
#         y = np.std(data[i])
#         means.append(x)
#         stds.append(y)

# def data_scale(self):
#     # for i in range(len(self.data)):
#     #     for j in range(len(self.data[i])):
#     #         if j < 120:
#     #             self.data[i][j] = (self.data[i][j] - self.min[0]) / (self.max[0] - self.min[0])
#     #         elif j < 240:
#     #             self.data[i][j] = (self.data[i][j] - self.min[1]) / (self.max[1] - self.min[1])
#     #         else:
#     #             self.data[i][j] = (self.data[i][j] - self.min[j - 238]) / (self.max[j - 238] - self.min[j - 238])
#     #
#     # return 0
#     Standard_data = StandardScaler().fit_transform(self.data)
#     print(Standard_data)
#     return Standard_data


if __name__ == '__main__':
    folder = Folder(index=4)
    print(folder.data.shape)
# print(len(folder))
# print(len(folder.data))
# for i in range(len(folder)):
#     print(len(folder.data[i]))
# d = folder.__getitem__(0)
# print('d', d[0])
# print('size', d[0].size())
