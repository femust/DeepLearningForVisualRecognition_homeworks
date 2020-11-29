import torch
import matplotlib.pyplot as plt
from dataloader import Dataset
from load_mnist import load_mnist

dataset_path = "dataset"
# test dataset
test_dataset = Dataset(dataset_path, "testing", range(10))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=48)

# defining network
num_class = 10
num_neurons = (50, 20)


def get_block(input_dim, output_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.ReLU(inplace=True),
    )


class FashionMNISTClassifier(torch.nn.Module):
    def __init__(self, in_feat, out_feat, num_neurons=(50, 10)):
        super(FashionMNISTClassifier, self).__init__()

        self.linears = torch.nn.ModuleList(
            [get_block(in_feat, num_neurons[0])])
        self.linears.extend([
            get_block(num_neurons[i], num_neurons[i + 1])
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
        loss.backward()
        optimizer.step()

    return epoch_loss


def get_accuracy(model, data):
    accuracy = 0
    for img, label in data:
        predict = model.forward(img)
        accuracy += torch.sum(
            torch.argmax(predict, axis=1) == label).item() / label.shape[0]

    return accuracy / data.__len__()


def ex_21a():

    accuracy = []

    # defining network
    model = FashionMNISTClassifier(28 * 28, num_class, num_neurons)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # training for certain epochs

    for class_ in range(10):
        train_dataset = Dataset(dataset_path, 'training', [class_])
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=48)
        epochs = 5
        for epoch in range(epochs):
            print("Epoch {}".format(epoch))
            _ = train(model, train_dataloader, optimizer, criterion)
            acc = get_accuracy(model, test_dataloader)
            accuracy.append(acc)

    return accuracy


def ex_21b():

    accuracy = []

    # defining network
    model = FashionMNISTClassifier(28 * 28, num_class, num_neurons)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # training for certain epochs
    classes = range(10)
    train_dataset = Dataset(dataset_path, 'training', classes)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=48,
                                                   shuffle=False)
    # without shuffle
    epochs = 25
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        _ = train(model, train_dataloader, optimizer, criterion)
        acc = get_accuracy(model, test_dataloader)
        accuracy.append(acc)

    return accuracy


def ex_21c():

    accuracy = []

    # defining network
    model = FashionMNISTClassifier(28 * 28, num_class, num_neurons)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # training for certain epochs
    classes = range(10)
    train_dataset = Dataset(dataset_path, 'training', classes)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=48,
                                                   shuffle=True)
    # with shuffle
    epochs = 25
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        _ = train(model, train_dataloader, optimizer, criterion)
        acc = get_accuracy(model, test_dataloader)
        accuracy.append(acc)

    return accuracy


if __name__ == '__main__':

    # exercise 2.1
    acc_a = ex_21a()
    acc_b = ex_21b()
    acc_c = ex_21c()
    plt.plot(acc_a[::2], '-^', label='sequential class')
    plt.plot(acc_b, '-^', label='without shuffling')
    plt.plot(acc_c, '-^', label='with shuffling')
    plt.show()
