import torch
from folders import Folder


class DataLoader(object):

    def __init__(self, batch_size=1, istrain=True, index=1):

        self.batch_size = batch_size
        self.istrain = istrain
        self.index = index

        # Train
        if istrain:
            self.data = Folder(train=True,index=self.index)
        # Test
        else:
            self.data = Folder(train=False,index=self.index)

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=False)
        return dataloader

if __name__ == '__main__':
    train_loader = DataLoader(64, istrain=True)
    test_loader = DataLoader(64, istrain=False)
    train_data = train_loader.get_data()
    test_data = test_loader.get_data()

