import torch
from load_mnist import load_mnist
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


class KNNCLassifier:
    def __init__(self, k, train_imgs, train_labels):
        self.k = k
        self.train_imgs = train_imgs.reshape(list(train_imgs.shape)[0], -1)
        self.train_labels = train_labels

    def forward(self, x):
        '''
        x = input tensor of images
        '''
        x = x.reshape(list(x.shape)[0], -1)
        distances = torch.sqrt((self.train_imgs**2).sum(axis=1)[:, None] +
                               (x**2).sum(axis=1)[None, :] -
                               self.train_imgs.matmul(x.T) * 2)
        similar_imgs, similar_indices = distances.topk(self.k,
                                                       largest=False,
                                                       axis=0)

        test_labels = []

        for i in range(list(x.shape)[0]):
            k_labels = self.train_labels[similar_indices[:, i]]
            test_label, _ = k_labels.mode(axis=0)
            test_labels.append(test_label)

        return torch.FloatTensor(test_labels), similar_indices

    def get_accuracy(self, k_list, test_imgs, test_labels):

        acc_list = []
        num_test_images = list(test_imgs.shape)[0]

        for k in k_list:
            self.k = k
            predict_labels, _ = self.forward(test_imgs)
            accuracy = (predict_labels == test_labels
                        ).sum().data.numpy().item() / num_test_images

            acc_list.append(accuracy)
            print('Accuracy for ' + str(k) + ' kNN is: ' + str(accuracy))

        return acc_list

    def cross_validation(self, fold, k_list):

        train_imgs_folds = torch.chunk(self.train_imgs, fold)
        train_labels_folds = torch.chunk(self.train_labels, fold)

        print(len(train_imgs_folds))

        accuracy = np.zeros(k_list[-1])

        for i in range(fold):
            index = list(range(fold))
            index.pop(i)
            print(index)
            val_imgs_ = train_imgs_folds[i].reshape(-1, 28, 28)
            val_labels_ = train_labels_folds[i]
            # no training involved in k-means just setting up train values for NN
            print(val_imgs_.shape)
            train_imgs_ = [train_imgs_folds[i] for l in index]
            train_labels_ = [train_labels_folds[i] for l in index]
            print(len(train_imgs_))
            print(torch.cat(train_imgs_).shape)
            self.train_imgs = torch.cat(train_imgs_)
            self.train_labels = torch.cat(train_labels_)
            print(self.train_imgs.shape)
            acc_list = self.get_accuracy(k_list, val_imgs_, val_labels_)
            accuracy += np.array(acc_list)

        plt.plot(k_list, accuracy / fold, '-o')
        plt.xlabel('k value')
        plt.ylabel('accuracy')
        plt.grid('on')
        plt.title('cross validation accuracy')
        plt.show()

    def plot_confusion_matrix(self, test_imgs, test_labels):
        predict_labels, _ = self.forward(test_imgs)
        conf_mat = confusion_matrix(test_labels.data.numpy(),
                                    predict_labels.data.numpy(),
                                    labels=range(0, 10))
        plt.matshow(conf_mat)
        plt.show()

    def plot(self, x):
        test_labels, similar_indices = self.forward(x)
        fig, ax = plt.subplots(list(x.shape)[0], self.k + 1)
        cols = ['test']
        cols.extend(range(1, list(x.shape)[0] + 1))
        for ax_, col in zip(ax[0], cols):
            ax_.set_title(col)
        [axi.set_axis_off() for axi in ax.ravel()]
        for i in range(list(x.shape)[0]):
            ax[i, 0].imshow(x[i].data.numpy().reshape(28, 28).astype('uint8'))
            plt.axis('off')
            for k in range(self.k):
                ax[i, k + 1].imshow(
                    self.train_imgs[similar_indices[k,
                                                    i]].data.numpy().reshape(
                                                        28,
                                                        28).astype('uint8'))

        plt.show()


if __name__ == '__main__':

    # get training set with 1000 samples : 100 per class
    dataset_path = "dataset"
    train_imgs, train_labels = load_mnist(dataset="training",
                                          path=dataset_path)

    # GET 100 PER CLASS

    # load test label and check for accuracy
    test_imgs, test_labels = load_mnist(dataset="testing", path=dataset_path)

    ##### TASK a.1 #####
    knn_classifier = KNNCLassifier(1, train_imgs[0:1000, :, :],
                                   train_labels[0:1000])
    klist = range(1, 6)
    accuracy_list = knn_classifier.get_accuracy(klist, test_imgs, test_labels)

    ##### TASK a.2 #####
    plot_test = test_imgs[:10, :, :]
    plot_test_labels = test_labels[:10]
    knn_classifier.k = 1
    knn_classifier.plot(plot_test)
    knn_classifier.k = 5
    knn_classifier.plot(plot_test)
    plot_test = test_imgs[:100, :, :]
    plot_test_labels = test_labels[:100]
    knn_classifier.plot_confusion_matrix(plot_test, plot_test_labels)

    ##### TASK a.3 #####
    k_list = range(1, 16)
    fold = 5
    knn_classifier.cross_validation(fold, k_list)
