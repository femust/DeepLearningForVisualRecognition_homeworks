import torch as nn
import matplotlib.pyplot as plt
from dataloader import Dataset
from load_mnist import load_mnist

def get_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )

class FashionMNISTClassifier(nn.Module):
    def __init__(self, in_feat, out_feat, num_neurons=(50,10)):
        super(FashionMNISTClassifier, self).__init__()
        #self.linear = nn.Linear(in_feat, out_feat)
        self.linear = nn.Sequential(
            get_block(in_feat, num_neurons[0]),
            get_block(num_neurons[0], num_neurons[1]),
            get_block(num_neurons[1], out_feat),
            nn.Softmax()
        )

    def forward(self, x):
        return nn.sigmoid(self.linear(x))


def train(model, dataloader, optimizer):
    epoch_loss = 0
    for iter, (img, label) in enumerate(dataloader):
        predict = model(img)
        # loss = nn.functional.mse_loss(predict,
        #                                     label.reshape(-1, 1),
        #                                     reduction='mean')
        loss = nn.functional.BCEWithLogitsLoss(predict,
                                            label.reshape(-1, 1),
                                            reduction='mean')

        epoch_loss += loss.item()
        if iter % 100 == 0:
            print("Iteration {} loss is: {:.4f}".format(iter, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return epoch_loss


if __name__ == '__main__':

    dataset_path = "dataset"
    train_dataset = Dataset(dataset_path, 'training')
    train_dataloader = nn.utils.data.DataLoader(train_dataset,
                                                   batch_size=48,
                                                   shuffle=True,
                                                   drop_last=False)

    # output dim is 1 for label prediction
    model = Network(28 * 28, 1)
    optimizer = nn.optim.SGD(model.parameters(), lr=0.005)

    # training for certain epochs
    epochs = 25
    losses = []
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        loss = train(model, train_dataloader, optimizer)
        losses.append(loss)

    plt.plot(losses, '-^')
    plt.title('loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Epoch Loss')
    plt.grid()
    plt.show()

    # report accuracy on test dataset
    test_dataset = Dataset(dataset_path, 'training')
    test_dataloader = nn.utils.data.DataLoader(
        test_dataset,
        batch_size=test_dataset.__len__(),
        shuffle=True,
        drop_last=False)
    for test_imgs, test_label in test_dataloader:
        test_predict = model.forward(test_imgs)
        # test_predict label to 0 or 1
        test_predict[nn.where(test_predict < 0.5)] = 0.0
        test_predict[nn.where(test_predict >= 0.5)] = 1.0
        accuracy = (test_predict == test_label.reshape(
            -1, 1)).sum().data.numpy().item() / test_dataset.__len__()

    print('Test accuracy is: {:.02f}'.format(accuracy))
