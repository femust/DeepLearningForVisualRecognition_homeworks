import torch
from load_mnist import load_mnist


class Dataset:
    def __init__(self, path, dataset):
        imgs, self.labels = load_mnist(dataset=dataset, path=path)
        self.imgs = imgs.reshape(list(imgs.shape)[0], -1)
        self.imgs, self.labels = self.dataFilter([0, 1])
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

    def dataFilter(self, valid_labels):
        id = torch.where((self.labels == valid_labels[0])
                         | (self.labels == valid_labels[1]))[0]
        return self.imgs[id, :], self.labels[id]


if __name__ == "__main__":

    dataset_path = "dataset"

    train_dataset = Dataset(dataset_path, "training")
    dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=48,
                                             shuffle=True,
                                             drop_last=False)

    for img, label in dataloader:
        print(img.size(), label.shape)
