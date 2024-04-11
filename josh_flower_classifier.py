'''
filename: josh_flower_classifer.py
description: This script includes the classes and functions necessary to 
    suppor the train.py and predict.py scripts for the udacity 'AI Programming
    with Python' nanodegree program for creating my own image classifier.
'''

## Imports
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter


## function to create network
def create_flower_network(arch, learn_rate, hidden_units):
    '''
    create_flower_network function will create a untrained network based on the 
    pre-trained VGG13 or RESNET34 pre-trained network with 
    an updated classifier layer for 102 possible classifications.

    Parameters
    ----------
    arch : string
        either "vgg" or "resnet".
    learn_rate : float
        to be used as the learning rate for the model.
    hidden_units : int
        number of hidden units above 102, but less than the 
            following based on the model:
                resnet: 512
                vgg: 25088

    Returns
    -------
    model : torchvision models vgg13 OR resnet34
        configured model for flower classification that is ready to train.
    optimizer : torch Adam optimizer
        optimizer ready to be used for training the model.
    criterion : torch nn.NLLLoss
        criterion ready to be used for training & testing the model.

    '''
    
    if arch == 'vgg':
        ## Set the model to the VGG13 model with default weights
        model = models.vgg13(weights='DEFAULT')
        
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        
        # Update the Classifier layer
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, hidden_units)),
                                  ('relu1', nn.ReLU()),
                                  ('do1', nn.Dropout(0.2)),
                                  ('fc2', nn.Linear(hidden_units, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))    
        model.classifier = classifier
        
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
        
    elif arch == 'resnet':
        ## Set the model to the Resnet34 model with default weights
        model = models.vgg13(weights='DEFAULT')
        
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        
        # Update the Classifier layer
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(512, hidden_units)),
                                  ('relu1', nn.ReLU()),
                                  ('do1', nn.Dropout(0.2)),
                                  ('fc2', nn.Linear(hidden_units, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))    
        model.fc = classifier
        
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.fc.parameters(), lr=learn_rate)
    else:
        print(f'invalid architecture {arch} provided. only "vgg" and "resnet" ',
              'are accepted.')

    
    # Set criterion to use NLLLoss()
    criterion = nn.NLLLoss()
    
    return model, optimizer, criterion
    
## Function to prepare flower data for training/testing/validating
def prepare_dataloaders(data_dir):
    '''
    prepare_data creates data loaders to be used in training and validating
    a model.

    Parameters
    ----------
    data_dir : string
        folder name in the same directory as this script where files are stored
        in subfolders of 'train', 'test' and 'valid' with subfolders with 
        numbers 1-102 representing the classes, in which the respective images
        are stored.

    Returns
    -------
    train_loader : Torch Dataloader
        Dataloader with all of the Training data.
    test_loader : Torch Dataloader
        Dataloader with all of the Testing data.
    valid_loader : Torch Dataloader
        Dataloader with all of the Validation data.
    class_to_idx : tbd
        Class to Index dictionary to be set with the training model

    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Transforms for Training, Test and Validation Datasets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    return train_loader, test_loader, valid_loader

## Function to run test data through a model
def testDataset(msg_str, loader, model, criterion):
    '''
    testDataset will test a provided model with a provided dataloader and will
    output the accuracy and loss, as well as printing a message related to 

    Parameters
    ----------
    msg_str : str
        Message string to be included in the printout to provide context for 
        the testDataset run.
    loader : Torch Dataloader
        Should be the "test" or "valid" data loader, not the "train" dataloader
    model : torchvision models vgg13 OR resnet34
        may be a pre-trained or untrained model.
    criterion : torch nn.NLLLoss
        criterion created when the model to be trained was created.

    Returns
    -------
    test_accuracy : float
        float of the accuracy of the model based on the provided data.
    test_loss: float
        the loss for the test run.

    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    accuracy = 0
    running_loss = 0
    model.eval()
    with torch.no_grad():
        #Set model to evaluation mode
        e_start = time.time()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model.forward(images)
            
            batch_loss = criterion(log_ps, labels)
            running_loss += batch_loss
    
            #calculate the accuracy
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        test_accuracy = accuracy/len(loader)*100
        test_loss = running_loss/len(loader)
    
        e_end = time.time()

        print(f'Model Performance for {msg_str}')
        print(f'Model Duration on Test Data (seconds): {e_end - e_start:.2f} ',
              f'Accuracy: {test_accuracy:.2f}% ',
              f'Loss: {test_loss}')
    
    model.train()
    return test_accuracy, test_loss
    
    
    
# Function for Trainig the model and testing it with each epoch
def trainModel(model, train_loader, test_loader, optimizer, criterion, epochs):
    '''
    trainModel will train a provided model with the provided inputs.

    Parameters
    ----------
    model : torchvision models vgg13 OR resnet34
        may be a pre-trained or untrained model.
    train_loader : Torch Dataloader
        Training data Dataloader.
    test_loader : Torch Dataloader
        Testing data Dataloader.
    optimizer : torch Adam optimizer
        optimizer created when the model was created.
    criterion : torch nn.NLLLoss
        criterion created when the model was created.
    epochs : int
        number of epochs to train the model through the data.

    Returns
    -------
    model : torchvision models vgg13 OR resnet34
        trained model.
    optimizer : torch Adam optimizer
        updated optimizer in case it is needed later for further training.

    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    running_loss = 0
    step_size = 20
    start = time.time() #start of training
    for e in range(epochs):
        t_start = time.time()
        for ii, (images, labels) in enumerate(train_loader):
            if ii % step_size == 0:
                b_start = time.time()
            
            images, labels = images.to(device), labels.to(device)
            
            log_ps = model.forward(images)
            train_loss = criterion(log_ps, labels)
            
            optimizer.zero_grad()
    
            train_loss.backward()
            optimizer.step()
            
            running_loss += train_loss.item()
            
            if (ii+1) % step_size == 0:
                b_end = time.time()
                b_set = f'{ii+2-step_size} to {ii+1}'
                print(f'Epoch: {e+1} ',
                      f'Training Batch: {b_set} ',
                      f'Duration: {b_end - b_start:.2f} ',
                      f'Loss: {running_loss/step_size:.6f}')
                testDataset(f'After Epoch {e+1}, Batch {b_set}', test_loader, 
                            model, criterion)
                running_loss = 0
              
        t_end = time.time()
        print(f'Training Epoch {e+1} duration: {t_end - t_start:.2f}')
        testDataset(f'After Training Epoch {e+1}', test_loader, model, criterion)
    
    end = time.time()
    print(f'Total Training Time (Seconds): {end - start:.2f}')
    return model, optimizer
    
    
# Run Validation Data through the model and Create a Model Checkpoint
def modelCheckpoint(model, optimizer, criterion, valid_loader, 
                    checkpoint_filename, learn_rate, hidden_units):
    '''
    modelCheckpoint will test the model using validation data, then save the
    model to a checkpoint file.

    Parameters
    ----------
    model : torchvision models vgg13 OR resnet34
        should be a fully trained model
    optimizer : torch Adam optimizer
        optimizer after it has been through training
    criterion : torch nn.NLLLoss
        criterion created when the model was created.
    valid_loader : Torch Dataloader
        Validation data Dataloader.
    checkpoint_filename : str
        string with the filename to use for the checkpoint file.
    learn_rate : float
        used to include in checkpoint file.
    hidden_units : int
        used to include in checkpoint file.

    Returns
    -------
    None.

    '''
    
    final_accuracy, final_loss = testDataset("Test with Validation Data", 
                                            valid_loader, model, criterion)

    checkpoint = {'model_type': str(type(model)),
                  'learn_rate': learn_rate,
                  'hidden_units': hidden_units,
                  'classifier_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'accuracy': final_accuracy,
                  'class_to_idx': model.class_to_idx}
    
    torch.save(checkpoint, checkpoint_filename)
    
    
    
    
    
    