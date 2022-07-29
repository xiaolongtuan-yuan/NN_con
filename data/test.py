import numpy as np
import pandas as pd

import xlwt, xlrd
from xlutils.copy import copy

df = np.array(pd.read_csv('data1.csv'))
print()

data = xlrd.open_workbook('data1.xls', formatting_info=True)
excel = copy(wb=data)  # 完成xlrd对象向xlwt对象转换
excel_table = excel.get_sheet(0)  # 获得要操作的页
table = data.sheets()[0]
nrows = table.nrows  # 获得行数
ncols = table.ncols  # 获得列数

shape = df.shape

for i in range(shape[0]):
    for j in range(shape[1]):
        excel_table.write(i+1, j, df[i][j])
excel.save('data2.xls')

# # 初始化最大值和最小值
        # # 把最大值设置的非常小，更容易遇到比它大的去更新最大值(因为不清楚各列的最大值到底多大，有的可能10e-5就是最大值了，因此需要把初始最大值设的非常小)
        # max_values = np.zeros(10) - 1e10
        # # 同理，把最小值设的非常大。
        # min_values = np.zeros(10) + 1e10
        #
        # # 分别更新每列的最大值和最小值
        # for i, key in enumerate(input_info):
        #     for j in range(len(data[key])):  # 遍历每一列
        #         # 更新第i列的最大值
        #         if data[key][j] > max_values[i]:
        #             max_values[i] = data[key][j]
        #         # 更新第i列的最小值
        #         if data[key][j] < min_values[i]:
        #             min_values[i] = data[key][j]
        #
        # # # 打印各列的最大最小值
        # print(max_values)
        # print(min_values)
        #
        # # record1 = 'max: '
        # # record2 = 'min: '
        # #
        # # max = max_values.tolist()
        # # min = min_values.tolist()
        #
        # # for i in range(10):
        # #     x1 = str(max[i])
        # #     record1 = record1 + x1 + "  "
        # #     x2 = str(min[i])
        # #     record2 = record2 + x2 + "  "
        # #
        # #
        # # file = open("./record.txt", "a")
        # # file.write(record1)
        # # file.write(record2)
        #
        # # 得到各列的最大最小值后，并应用缩放公式对各列数据进行特征缩放
        # for i, key in enumerate(input_info):
        #     for j in range(len(data[key])):
        #         data[key][j] = (data[key][j] - min_values[i]) / (max_values[i] - min_values[i])




#
    #     self.train = train
    #
    #     # 打开excel
    #     wb = xlrd.open_workbook('data/data2.xls')
    #     # 按工作簿定位工作表
    #     sh = wb.sheets()[0]
    #
    #     data_dict = dict()
    #     train_dict = dict()
    #     test_dict = dict()
    #     for title in sh.row_values(0):
    #         data_dict[title] = []  # 读取第一行的标题, 每个标题作为data_dict的一个键, 初始化各个键的值为空列表
    #         train_dict[title] = []
    #         test_dict[title] = []
    #     # index_dict = {0: "x1", 1: "x2", 2: "x3", 3: "x4", 4: "x5", 5: "x6", 6: "y1", 7: "y2", 8: "y3", 9: "y4"}
    #     # key_list = ["x1", "x2", "x3", "x4", "x5", "x6", "y1", "y2", "y3", "y4"]
    #
    #     # 当模型只预测但指标时
    #     index_dict = {0: "x1", 1: "x2", 2: "x3", 3: "x4", 4: "x5", 5: "x6", 6: "y1"}
    #     key_list = ["x1", "x2", "x3", "x4", "x5", "x6", "y1"]
    #
    #     intervel = 0  # 当预测y2时，改为1
    #     # 载入数据
    #     row_num = sh.nrows
    #     col_num = 7
    #     for row_i in range(1, row_num):
    #         for col_i in range(col_num):
    #             data_dict[index_dict[col_i]].append(sh.cell(row_i, col_i + intervel).value)  # 字典总数据10列 n行
    #
    #     # 数据归一化
    #     scaled_data_dict = self.data_scale(data_dict, key_list)
    #     # scaled_data_dict = data_dict  # 不进行归一化
    #
    #     # 划分训练集和测试集
    #     self.index = list(range(row_num - 1))
    #     random.shuffle(self.index)
    #     train_index = self.index[0:int(round(0.9 * len(self.index)))]  # 训练集采用前80%数据
    #     test_index = self.index[int(round(0.9 * len(self.index))):len(self.index)]  # 后20%测试集
    #
    #     for row_i in range(len(self.index)):
    #         if row_i in test_index:
    #             for col_i in range(col_num):
    #                 test_dict[index_dict[col_i]].append(scaled_data_dict[index_dict[col_i]][row_i])
    #         else:
    #             for col_i in range(col_num):
    #                 train_dict[index_dict[col_i]].append(scaled_data_dict[index_dict[col_i]][row_i])
    #
    #     # 扩充维度
    #     for key, values in train_dict.items():
    #         train_dict[key] = np.expand_dims(np.array(values), axis=1)
    #         # 例如将[1,2,3,4,5,6]扩展为[[1],[2],[3],[4],[5],[6]]
    #     for key, values in test_dict.items():
    #         test_dict[key] = np.expand_dims(np.array(values), axis=1)
    #
    #     input_key = ["x1", "x2", "x3", "x4", "x5", "x6"]
    #     output_key = ["y1", "y2", "y3", "y4"]
    #     if self.train:
    #         self.input = np.concatenate((train_dict[input_key[0]],
    #                                      train_dict[input_key[1]],
    #                                      train_dict[input_key[2]],
    #                                      train_dict[input_key[3]],
    #                                      train_dict[input_key[4]],
    #                                      train_dict[input_key[5]]), axis=1)  # 将xi连接为一个数组
    #         self.label = train_dict[output_key[0]]  # y1
    #         #                            train_dict[output_key[1]],
    #         #                            train_dict[output_key[2]],
    #         #                            train_dict[output_key[3]]
    #         self.input = torch.from_numpy(self.input)  # 将数组转化为张量
    #         self.label = torch.from_numpy(self.label)
    #     else:
    #         self.input = np.concatenate((test_dict[input_key[0]],
    #                                      test_dict[input_key[1]],
    #                                      test_dict[input_key[2]],
    #                                      test_dict[input_key[3]],
    #                                      test_dict[input_key[4]],
    #                                      test_dict[input_key[5]]), axis=1)
    #         self.label = test_dict[output_key[0]]  # y1
    #         self.input = torch.from_numpy(self.input)
    #         self.label = torch.from_numpy(self.label)
    #     print(self.input.shape[0])
    #     print(self.label.shape[0])