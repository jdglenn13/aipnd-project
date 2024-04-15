'''
filename: josh_flower_classifer.py
description: This script includes the functions necessary to 
    support the train.py and predict.py scripts for the udacity 'AI Programming
    with Python' nanodegree program for creating my own image classifier.
author: Joshua Glenn (jglenn@its.jnj.com)
last modified: 12Apr2024
** _udacity script created to work int he udacity python 3.6 environment
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
        model = models.vgg13(pretrained=True)
        
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
        model = models.resnet34(pretrained=True)
        
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
    class_to_idx : Dictionary
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
    
    class_to_idx = train_data.class_to_idx
    
    return train_loader, test_loader, valid_loader, class_to_idx

## Function to run test data through a model
def testDataset(msg_str, loader, model, criterion, device):
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
    device : torch.device()
        set to either 'cpu' or 'cuda'.

    Returns
    -------
    test_accuracy : float
        float of the accuracy of the model based on the provided data.
    test_loss: float
        the loss for the test run.

    '''
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
              f'Loss: {test_loss:.3f}')
    
    model.train()
    return test_accuracy, test_loss
    
    
    
# Function for Trainig the model and testing it with each epoch
def trainModel(model, train_loader, test_loader, optimizer, criterion, epochs,
               device):
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
    device : torch.device()
        set to either 'cpu' or 'cuda'.

    Returns
    -------
    model : torchvision models vgg13 OR resnet34
        trained model.
    optimizer : torch Adam optimizer
        updated optimizer in case it is needed later for further training.

    '''
    
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
                            model, criterion, device)
                running_loss = 0
              
        t_end = time.time()
        print(f'Training Epoch {e+1} duration: {t_end - t_start:.2f}')
        testDataset(f'After Training Epoch {e+1}', test_loader, model, 
                    criterion, device)
    
    end = time.time()
    print(f'Total Training Time (Seconds): {end - start:.2f}')
    return model, optimizer
    
    
# Run Validation Data through the model and Create a Model Checkpoint
def modelCheckpoint(model, optimizer, criterion, valid_loader, 
                    checkpoint_filename, learn_rate, hidden_units, arch,
                    device):
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
    arch : string
        either "vgg" or "resnet".
    device : torch.device()
        set to either 'cpu' or 'cuda'.

    Returns
    -------
    None.

    '''
    
    final_accuracy, final_loss = testDataset("Test with Validation Data", 
                                            valid_loader, model, criterion,
                                            device)

    checkpoint = {'arch': arch,
                  'learn_rate': learn_rate,
                  'hidden_units': hidden_units,
                  'classifier_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'accuracy': final_accuracy,
                  'class_to_idx': model.class_to_idx}
    
    torch.save(checkpoint, checkpoint_filename)
    

# Load Model function
def load_checkpoint(filepath, device):
    '''
    load_checkpoint will take a filepath to a checkpoint file and load the 
    model, optimizer and criterion.

    Parameters
    ----------
    filepath : str
        filepath to the checkpoint file.
    device : torch.device()
        set to either 'cpu' or 'cuda'.

    Returns
    -------
    model : torchvision models vgg13 OR resnet34
        configured model for flower classification that is already trained
    optimizer : torch Adam optimizer
        optimizer previously used to train the model and ready for further 
        training
    criterion : torch nn.NLLLoss
        criterion ready to be used for training & testing the model.

    '''
    checkpoint = torch.load(filepath)
    
    arch = checkpoint['arch']
    learn_rate = checkpoint['learn_rate']
    hidden_units = checkpoint['hidden_units']
    
    model, optimizer, criterion = create_flower_network(arch, 
                                                        learn_rate, 
                                                        hidden_units)
    

    model.load_state_dict(checkpoint['classifier_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    accuracy = checkpoint['accuracy']
    
    print(f'"{filepath}" "{arch}"model loaded with '
          f'accuracy of {accuracy:.2f}%')
    
    return model, optimizer, criterion    
    
## process_image function
def process_image(image):
    '''
    process_image will take an input image file and prepare the image for 
    classification through a trained model, including normalization of the 
    image.

    Parameters
    ----------
    image : str
        path/filename to image file to be processed.

    Returns
    -------
    np_image : np.array()
        nparray representing the input image.

    '''
    
    # Process a PIL image for use in a PyTorch model
    size = (256, 256)
    crop = 224
    with Image.open(image) as img:
        img = img.resize(size)
        width, height = img.size   # Get dimensions
    
        # set new crop dimensions
        left = (width - crop)/2
        top = (height - crop)/2
        right = (width + crop)/2
        bottom = (height + crop)/2
        
        # Crop the center of the image
        img = img.crop((left, top, right, bottom))
        np_image = np.array(img)
    
        #Normalize and transform
        #help from the following page
        # --> https://stackoverflow.com/questions/65617755/how-to-replicate-pytorch-normalization-in-opencv-or-numpy
        MEAN = 255 * np.array([0.485, 0.456, 0.406])
        STD = 255 * np.array([0.229, 0.224, 0.225])
        np_image = np_image.transpose(-1, 0, 1)
        np_image = (np_image - MEAN[:, None, None]) / STD[:, None, None]

    return np_image


## function to display both the image and the plot of the prediction
def predict_show(image, top_class, act_class, probs, 
                 image_path):
    '''
    function used to plot both the image and the prediction called from the 
    predict function.

    Parameters
    ----------
    image : np.array()
        np.array of PIL image created from process_image function.
    top_class : list
        list of the top classes.
    act_class : str
        actual class of the provided image (1-102).        
    probs : list
        list of the probabilities to plot
    image_path : str
        path of the image that was processed.

    Returns
    -------
    None.

    '''
    
    ##Establish the plots
    fig, ax = plt.subplots(2,1,figsize=[10,10])
    fig.subplots_adjust(left=0.3)

    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax[0].set_title(image_path)
        
    ax[0].imshow(image)
    
    ## Plot the probabilities

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Example data
    class_labels = list(itemgetter(*top_class)(cat_to_name))
    if act_class != '0':
        for i in range(len(class_labels)):
            if cat_to_name[act_class] == class_labels[i]:
                class_labels[i] += '**' 
    
    y_pos = np.arange(len(class_labels))
    
    ax[1].barh(y_pos, probs, align='center')
    ax[1].set_yticks(y_pos, labels=class_labels)
    ax[1].invert_yaxis()  # labels read top-to-bottom
    ax[1].set_xlabel('Probability')
    if act_class != '0':    
        ax[1].set_ylabel('Top5 Classes (** is the actual class)')
    else:
        ax[1].set_ylabel('Top5 Classes')
        
    ax[1].set_title(f'Prediction for {image_path}')
    plt.show()    


## imshow function to display an image after it has been through process_image
def imshow(image, title=None):
    '''
    Displays the image that has been processed

    Parameters
    ----------
    image : np.array()
        np.array of image processed through process_image().
    title : str, optional
        Title for the plot

    Returns
    -------
    none

    '''
    fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.set_title(title)
        
    ax.imshow(image)



    


def predict(image_path, act_class, model, device, topk=5, visualize=False):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.

    Parameters
    ----------
    image_path : str
        path to the image file to be classified.
    act_class : str
        actual class of the provided image (1-102).
    model : torchvision models vgg13 OR resnet34
        configured model for flower classification that is already trained
    device : torch.device()
        set to either 'cpu' or 'cuda'.
    topk : int, optional
        Top k probabilities.  The default is 5.
    visualize : boolean
        The default is False with the expectation that the results will be 
        displayed in text form from the command line output.  True will plot
        the image and the probabilities using matplotlib.
        
        

    Returns
    -------
    None.

    '''
    model.to(device)
    
    # Predict the class of the image
    image = torch.tensor(process_image(image_path)).float().to(device)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        model.eval()
        log_ps = model.forward(image)
    
        #calculate the accuracy
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(topk)
        #invert the class index
        class_to_idx = {v: k for k, v in model.class_to_idx.items()}
        top_class = list(itemgetter(*top_class.to('cpu').reshape(-1).tolist())(class_to_idx))
        model.train()
        
    probs = top_p.to('cpu').reshape(-1).tolist()

    if visualize:
        predict_show(process_image(image_path), top_class, act_class, probs, 
                     image_path)
    else:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
        
        class_labels = list(itemgetter(*top_class)(cat_to_name))
        if act_class != '0':
            for i in range(len(class_labels)):
                if cat_to_name[act_class] == class_labels[i]:
                    class_labels[i] += '**' 
        
        if act_class != '0':    
            print('Top Classes are (** is the actual class):')
        else:
            print('Top Classes are:')
        
        for i in range(len(class_labels)):
            label = class_labels[i]
            prob = probs[i]*100
            print(f'Label: {label:20}   Probability: {prob:.2f}%')
    

