import torch
import matplotlib.pyplot as plt
from dataloader import Dataset
from load_mnist import load_mnist
import numpy as np

dataset_path = "dataset"
# test dataset
test_dataset = Dataset(dataset_path, "testing", range(10))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

# defining network
num_class = 10
num_neurons = (50, 20)


def get_block(input_dim, output_dim, activation):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        activation(),
    )


class FashionMNISTClassifier(torch.nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_neurons=(50, 10),
                 activation=torch.nn.ReLU):
        super(FashionMNISTClassifier, self).__init__()

        self.linears = torch.nn.ModuleList(
            [get_block(in_feat, num_neurons[0], activation)])
        self.linears.extend([
            get_block(num_neurons[i], num_neurons[i + 1], activation)
            for i in range(len(num_neurons) - 1)
        ])
        self.linears.append(torch.nn.Linear(num_neurons[-1], out_feat))

    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
        return x


def train(model, dataloader, optimizer, criterion):
    epoch_loss = 0
    for iter, (img, label) in enumerate(dataloader):
        predict = model(img)
        loss = criterion(predict, label)

        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    return epoch_loss


def get_accuracy(model, data):
    accuracy = 0
    for img, label in data:
        predict = model.forward(img)
        accuracy += torch.sum(
            torch.argmax(predict, axis=1) == label).item() / label.shape[0]

    return accuracy / data.__len__()


def sequential_class():

    accuracy = []

    # defining network
    model = FashionMNISTClassifier(28 * 28, num_class, num_neurons)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # training for certain epochs

    for class_ in range(10):
        train_dataset = Dataset(dataset_path, 'training', [class_])
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=64)
        epochs = 5
        for epoch in range(epochs):
            _ = train(model, train_dataloader, optimizer, criterion)
            acc = get_accuracy(model, test_dataloader)
            accuracy.append(acc)

    return accuracy


def without_shuffling():

    accuracy = []

    # defining network
    model = FashionMNISTClassifier(28 * 28, num_class, num_neurons)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # training for certain epochs
    classes = range(10)
    train_dataset = Dataset(dataset_path, 'training', classes)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=64,
                                                   shuffle=False)
    # without shuffle
    epochs = 25
    for epoch in range(epochs):
        _ = train(model, train_dataloader, optimizer, criterion)
        acc = get_accuracy(model, test_dataloader)
        accuracy.append(acc)

    return accuracy


def with_shuffling():

    accuracy = []

    # defining network
    model = FashionMNISTClassifier(28 * 28, num_class, num_neurons)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # training for certain epochs
    classes = range(10)
    train_dataset = Dataset(dataset_path, 'training', classes)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=64,
                                                   shuffle=True)
    # with shuffle
    epochs = 25
    for epoch in range(epochs):
        _ = train(model, train_dataloader, optimizer, criterion)
        acc = get_accuracy(model, test_dataloader)
        accuracy.append(acc)

    return accuracy


def cross_validation(func):

    # 5 fold cross validation
    fold = 5
    train_imgs, train_labels = load_mnist("training", dataset_path)
    train_imgs_folds = torch.chunk(train_imgs, fold)
    train_labels_folds = torch.chunk(train_labels, fold)

    accuracy = np.zeros(10)

    for i in range(fold):

        # defining network
        model = FashionMNISTClassifier(28 * 28, num_class, num_neurons, func)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        index = list(range(fold))
        index.pop(i)

        # validation set
        val_imgs_ = train_imgs_folds[i]
        val_labels_ = train_labels_folds[i]
        val_dataset = Dataset(None, None, range(10), val_imgs_, val_labels_)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=64,
                                                     shuffle=True)

        # get training set
        train_imgs_ = [train_imgs_folds[i] for l in index]
        train_labels_ = [train_labels_folds[i] for l in index]
        train_imgs = torch.cat(train_imgs_)
        train_labels = torch.cat(train_labels_)
        train_dataset = Dataset(None, None, range(10), train_imgs,
                                train_labels)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=64,
                                                       shuffle=True)

        # with shuffle
        epochs = 10
        for epoch in range(epochs):
            _ = train(model, train_dataloader, optimizer, criterion)
            acc = get_accuracy(model, val_dataloader)
            accuracy[epoch] += acc

    test_acc = get_accuracy(model, test_dataloader)

    return accuracy / fold, test_acc


class Swish(torch.nn.Module):
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta * torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


if __name__ == '__main__':

    # exercise 2.1
    print('Training for each class sequentially . . .')
    acc_a = sequential_class()
    print('Training without shuffling of the data . . .')
    acc_b = without_shuffling()
    print('Training with shuffling of the data . . .')
    acc_c = with_shuffling()
    plt.plot(acc_a[::2], '-^', label='sequential class')
    plt.plot(acc_b, '-^', label='without shuffling')
    plt.plot(acc_c, '-^', label='with shuffling')
    plt.legend()
    plt.grid()
    plt.show()

    # exercise 2.2
    actv_func = [
        torch.nn.Identity, torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh,
        torch.nn.LeakyReLU, Swish
    ]

    for function in actv_func:
        val_acc, test_accuracy = cross_validation(function)
        plt.plot(val_acc, '-^', label=str(function))
        print('The test accuracy for ' + str(function) + 'is: ' +
              str(test_accuracy))

    plt.legend()
    plt.grid()
    plt.show()
