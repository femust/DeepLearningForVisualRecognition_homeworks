import torch
from load_mnist import load_mnist


class Dataset:
    def __init__(self, path, dataset, classes, imgs=None, labels=None):
        if (path):
            imgs, labels = load_mnist(dataset=dataset, path=path)
        self.imgs = imgs.reshape(list(imgs.shape)[0], -1)
        self.labels = labels.type(torch.LongTensor)
        self.imgs, self.labels = self.dataFilter(classes)
        self.mean = self.getMean()
        self.std = self.getStd()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        img = self.imgs[index]
        label = self.labels[index]

        # data standardisation
        img = (img - self.mean) / self.std
        # data scaling
        # img = img / torch.max(torch.abs(self.imgs))

        return img, label

    def getMean(self):
        mean = torch.mean(self.imgs)
        return mean

    def getStd(self):
        std = torch.std(self.imgs)
        return std

    def dataFilter(self, classes):
        id = torch.tensor([], dtype=torch.long)
        for class_ in classes:
            idc = torch.where(self.labels == class_)[0]
            id = torch.cat((id, idc))
        return self.imgs[id, :], self.labels[id]
