import numpy as np
import time, os

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from myImageNN import MyCNN, MyFullyConnectedNN, MyLogRegNN


def writeHistoryPlots(history, modelType, filePath):
    history = np.array(history)
    plt.clf()
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig(filePath + modelType + '_loss_curve.png')
    plt.clf()
    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(filePath + modelType + '_accuracy_curve.png')



def LoadDataSet() -> (int, DataLoader, int, DataLoader):
    dataset = '../data/horse_no_horse/'

    bs = 32
    # TODO: Load image dataset
    # Hint: see the Transer_learning notebook on how this can be done

    # mean, std_dev = mean_and_standard_dev(train_data_loader)
    # print(f"Training data mean: {mean}, std-dev: {std_dev}")
    train_directory = os.path.join(dataset, 'train')
    valid_directory = os.path.join(dataset, 'valid')
    image_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }
    transforming = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=transforming),
        'valid': datasets.ImageFolder(root=valid_directory, transform=transforming)
    }
    train_data_size = len(data['train'])
    valid_data_size = len(data['valid'])

    train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True)
    valid_data_loader = DataLoader(data['valid'], batch_size=bs, shuffle=True)

    return train_data_size, train_data_loader, valid_data_size, valid_data_loader


def train_and_validate(myModel, criterion, optimizer, epochs=25):
    '''
    Function to train and validate
    Parameters
        :param myModel: Model to train and validate
        :param criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)

    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''
    train_data_size, train_data_loader, valid_data_size, valid_data_loader = LoadDataSet()
    history = []
    # TODO: Train model and validate model on validation set after each epoch
    for epoch in range(epochs):
        myModel.train()
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        for i, (inputs, labels) in enumerate(train_data_loader):
            optimizer.zero_grad()
            if myModel.type == 'cnn':
                outputs = myModel(inputs)
            else:
                outputs = myModel(inputs.view(-1, inputs.size(0)))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions[0].eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)
            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))
        with torch.no_grad():
            myModel.eval()
            for j, (inputs, labels) in enumerate(valid_data_loader):
                if myModel.type == 'cnn':
                    outputs = myModel(inputs)
                else:
                    outputs = myModel(inputs.view(-1, inputs.size(0)))
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)
                print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            avg_train_loss = train_loss / train_data_size
            avg_train_acc = train_acc / train_data_size
            avg_valid_loss = valid_loss / valid_data_size
            avg_valid_acc = valid_acc / valid_data_size

            history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

            print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%".format(
                    epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100))
    return myModel, history


def logreg_train_test():
    global criterion, optimizer
    # # TODO: train and test a logistic regression classifier implemented as a neural network
    print('##########################')
    print('Testing Logistic Regression')
    logRegModel = MyLogRegNN()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')  # Cost function - torch.nn.XXX loss functions
    optimizer = torch.optim.Adam(logRegModel.parameters(), 0.0001)  # Optimizer algorithm - torch.optim.XXX function
    finallogRegmodel, logRegHistory = train_and_validate(logRegModel, criterion, optimizer, epochs=20)
    writeHistoryPlots(logRegHistory, 'logRegModel', 'output/')


def fully_connected_train_test():
    global criterion, optimizer
    # TODO: train and test the fully connected DNN
    print('##########################')
    print('Testing Deep Neural Net')
    dnnModel = MyFullyConnectedNN()
    criterion = torch.nn.CrossEntropyLoss()  # Cost function - torch.nn.XXX loss functions
    optimizer = torch.optim.Adam(dnnModel.parameters(), 0.0001)  # Optimizer algorithm - torch.optim.XXX function
    finalDNNmodel, dnnHistory = train_and_validate(dnnModel, criterion, optimizer, epochs=20)
    writeHistoryPlots(dnnHistory, 'dnnModel', 'output/')


def cnn_train_test():
    global criterion, optimizer
    # TODO: train and test a CNN
    print('##########################')
    print('Testing Convolutional Neural Net')
    cnnModel = MyCNN()
    criterion = torch.nn.CrossEntropyLoss()  # Cost function - torch.nn.XXX loss functions
    optimizer = torch.optim.Adam(cnnModel.parameters())  # Optimizer algorithm - torch.optim.XXX function
    finalCNNmodel, cnnHistory = train_and_validate(cnnModel, criterion, optimizer, epochs=20)
    writeHistoryPlots(cnnHistory, 'cnnModel', 'output/')


if __name__ == '__main__':
    logreg_train_test()

    fully_connected_train_test()

    cnn_train_test()
