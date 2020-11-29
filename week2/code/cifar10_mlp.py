import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

input_size = 3 * 32 * 32
output_size = 10
no_hidden_nodes = 256
learning_rate = 0.01
num_epochs = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32
verbose = True


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    torch.manual_seed(43)
    val_size = 5000
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(val_dataset, batch_size=128, num_workers=4, pin_memory=True)
    return train_loader, test_loader, validation_loader


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(input_size, no_hidden_nodes)
        self.fc2 = nn.Linear(no_hidden_nodes, no_hidden_nodes)
        self.fc3 = nn.Linear(no_hidden_nodes, output_size)

    def forward(self, xb):
        out = xb.view(-1, input_size)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return F.log_softmax(out, dim=-1)


def display_log(idx, length, epoch_loss, acc, mode):
    if length >= 250:
        update_size = int(length / 250)
    else:
        update_size = 5

    if idx % update_size == 0 and idx != 0:
        # print ('=', end="")
        finish_rate = idx / length * 100
        print("\r   {} progress: {:.2f}%  ......  loss: {:.4f} , acc: {:.4f}".
              format(mode, finish_rate, epoch_loss / idx, acc), end="", flush=True)


def train_per_epoch(model, loss_fn, train_loader, optimizer, verbose):
    model.train()
    # initialize loss
    running_loss = 0.0
    accuracy = 0.0
    total = 0
    for batch_id, (images, labels) in enumerate(train_loader):
        images, labels = images.reshape(-1, input_size).to(device), labels.to(device)
        # zero the gradients
        optimizer.zero_grad()
        # Forward Pass
        outputs = model(images)

        # training accuracy
        _, predicted = torch.max(outputs.data, dim=1)
        correct = (predicted == labels).sum().item()
        accuracy += correct
        total += labels.size(0)

        # compute the loss
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()
        # Compute the gradient
        loss.backward()
        # update the weights
        optimizer.step()
        if verbose:
            display_log(batch_id, len(train_loader), running_loss, accuracy / total, 'training')

    accuracy = accuracy / total
    print('')
    return running_loss / len(train_loader), accuracy


def validation_per_epoch(model, loss_fn, validation_loader, optimizer, verbose):
    # validation phase
    model.eval()
    running_loss = 0.0
    accuracy = 0
    total = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(validation_loader):
            images, labels = images.reshape(-1, input_size).to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, dim=1)
            correct = (predicted == labels).sum().item()
            total += labels.size(0)
            accuracy += correct
            if verbose:
                display_log(idx, len(validation_loader), running_loss, accuracy / total, 'validation')
        accuracy = accuracy / total
    print("")
    return running_loss / len(validation_loader), accuracy


def evaluate(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def visualize(metrics):
    fig, ax = plt.subplots(2)

    epochs = np.linspace(1, num_epochs, num_epochs).astype(int)
    ax[0].plot(epochs, metrics['train_loss'], label="Training loss")
    ax[0].plot(epochs, metrics['val_loss'], label="validation loss", axes=ax[0])
    ax[0].set_xlabel('num of epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_xticks(epochs)
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(epochs, metrics['train_accuracy'], label="Training accuracy")
    ax[1].plot(epochs, metrics['val_accuracy'], label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    ax[1].set_xlabel('num of epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xticks(epochs)
    plt.show()


def main():
    cuda = torch.cuda.is_available()
    train_loader, test_loader, validation_loader = load_data()
    model = MLPNet().to(device)
    print("==============================================================")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    metrics = {}
    metrics['train_loss'] = []
    metrics['val_loss'] = []
    metrics['train_accuracy'] = []
    metrics['val_accuracy'] = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        train_loss, train_acc = train_per_epoch(model, loss_fn, train_loader, optimizer, verbose)
        metrics['train_loss'].append(train_loss)
        metrics['train_accuracy'].append(train_acc)
        val_loss, val_acc = validation_per_epoch(model, loss_fn, validation_loader, optimizer, verbose)
        metrics['val_loss'].append(val_loss)
        metrics['val_accuracy'].append(val_acc)
        print('\n        Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(train_loss, val_loss))
        print('        Training acc: {:.4f},  Validation acc: {:.4f}\n'.format(train_acc, val_acc))

    evaluate(model, test_loader)
    visualize(metrics)


if __name__ == '__main__':
    main()
