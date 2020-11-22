import numpy as np
import torch
from load_mnist import load_mnist

DATA_PATH = "dataset"

class Dataset:
    def __init__(self, dataset):
        self.imgs, self.labels = load_mnist(dataset=dataset, path=DATA_PATH)
        self.imgs, self.labels = self.dataFilter([1,2])
        """self.mean = self.getMean()
        self.std = self.getStd()"""
    
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        return ((self.imgs[index]/127.5-1.to(torch.float32)), self.labels[index].item())
    
    """def getMean(self):
        mean = 0.0
        for i in range(self.__len__()):
            mean += torch.mean(self.imgs[i])
        return mean / self.__len__()
    
    def getStd(self):
        std = 0.0
        for i in range(self.__len__()):
            std += torch.std(self.imgs[i])
        return std / self.__len__()
            """
    def dataFilter(self, valid_labels):
        new_imgs = []
        new_labels = []
        
        for i in range(self.__len__()):
            if self.labels[i] in valid_labels:
                # without .numpy() --> ValueError: only one element tensors can be converted to Python scalars
                new_imgs.append(self.imgs[i].numpy())
                new_labels.append(self.labels[i])                
        return torch.tensor(new_imgs), torch.tensor(new_labels)


if __name__ == "__main__":    
    train_dataset = Dataset("training")
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)

    for img, label in dataloader:
        print(img.size())
        break
#test_dataset = Dataset("testing")