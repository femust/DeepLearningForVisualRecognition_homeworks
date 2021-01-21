import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import pickle
from pathlib import Path 

input_size = 3 * 32 * 32
output_size = 10
no_hidden_nodes = 512
num_epochs = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
verbose = False

class CIFAR10Dataset:
    def __init__(self, dataset='train', path='.'):
        path = Path(path)
        if dataset == 'train':
            batches = [self.load_batch(p) for p in path.glob('data_batch_*')]
            self.images = torch.cat([b for b, _ in batches])
            self.labels = torch.cat([b for _, b in batches])
        elif dataset == 'test':
            self.images, self.labels = self.load_batch(path / 'test_batch')
            
        # normalize
        self.images = self.images.float()
        self.labels = self.labels.long()
        
        std, mean = torch.std_mean(self.images, dim=0)
        self.images = (self.images - mean) / std
        
        assert len(self.images) == len(self.labels)
    
    def load_batch(self, path):
        with open(path, 'rb') as file:
            data_dict = pickle.load(file, encoding='bytes')
        images = data_dict[b'data'].reshape((-1, 32, 32, 3), order='F').swapaxes(1,2)
        images = torch.from_numpy(images)
        labels = torch.Tensor(data_dict[b'labels'])
        return images, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]

class CUDADataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        
    def __iter__(self):
        for images, labels in self.dataloader:
            yield [images.to('cuda:0'), labels.to('cuda:0')]
            
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
        return F.softmax(out, dim=-1)


def train_per_epoch(model, loss_fn, train_loader, optimizer):
    model.train()
    # initialize loss
    running_loss = 0.0
    accuracy = 0.0
    total = num_batches = 0
    for images, labels in train_loader:
        # zero the gradients
        optimizer.zero_grad()
        # Forward Pass
        outputs = model(images)
        # training accuracy
        _, predicted = torch.max(outputs.data, dim=1)
        correct = (predicted == labels).sum().item()
        accuracy += correct
        total += labels.size(0)
        num_batches += 1
        # compute the loss
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()
        # Compute the gradient
        loss.backward(retain_graph=True)
        # update the weights
        optimizer.step()
    accuracy = accuracy / total
    return running_loss / num_batches, accuracy


def get_loss(model, test_loader, loss_fn):
    loss_sum = num_batches = 0
    for image_batch, label_batch in test_loader:
        predictions = model(image_batch)
        loss = loss_fn(predictions, label_batch)
        loss_sum += loss.item()
        num_batches += 1

    return loss_sum / num_batches


def validation_per_epoch(model, loss_fn, validation_loader):
    # validation phase
    model.eval()
    running_loss = 0.0
    num_batches  = 0
    accuracy = 0
    total = 0
    with torch.no_grad():
        for images, labels in validation_loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            correct = (predicted == labels).sum().item()
            total += labels.size(0)
            num_batches += 1
            accuracy += correct
        accuracy = accuracy / total
    return running_loss / num_batches


def evaluate(model, test_dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def load_data():
    use_gpu = True
    
    train_dataloader = DataLoader(CIFAR10Dataset('train','/home/ramsay/s7ravenk/dlvr/DeepLearningForVisualRecognition_homeworks/week4/code/data/cifar_10'), batch_size=64, shuffle=True, pin_memory=use_gpu)
    test_dataloader = DataLoader(CIFAR10Dataset('test','/home/ramsay/s7ravenk/dlvr/DeepLearningForVisualRecognition_homeworks/week4/code/data/cifar_10'), batch_size=1_024, pin_memory=use_gpu)
    if use_gpu:
        train_dataloader = CUDADataLoader(train_dataloader)
        test_dataloader = CUDADataLoader(test_dataloader)
    return train_dataloader, test_dataloader

def get_optimizer(train_dataloader, test_dataloader, lr=None, verbose = False):    
    model = MLPNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metrics = {}
    metrics['train_loss'] = []
    metrics['val_loss'] = []
    metrics['train_accuracy'] = []
    metrics['test_accuracy'] = []
    for epoch in range(num_epochs):
        train_loss, train_acc = train_per_epoch(model, loss_fn, train_dataloader, optimizer)
        metrics['train_loss'].append(train_loss)
        metrics['train_accuracy'].append(train_acc)
        val_loss = validation_per_epoch(model, loss_fn, test_dataloader)
        metrics['val_loss'].append(val_loss)
        acc = evaluate(model, test_dataloader)
        metrics['test_accuracy'].append(acc)
        if verbose and epoch % 5 == 0:
            print(f"epoch[{epoch}] acc: {100 * acc:.2f}% train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}")
    return model, metrics 
