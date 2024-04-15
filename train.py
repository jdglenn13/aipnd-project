'''
filename: train.py
description: train.py is intended to facilitate training a new model using 
        either the vgg or resnet pre-trained models against an input data.
author: Joshua Glenn (jglenn@its.jnj.com)
last modified: 11Apr2024

INPUTS: Arguments to the train.py script
    --image_directory: Directory with flower images to be used for training, 
        testing and validation, already organized as follows: 
            <image_directory>/<train/test/valid>/<class # 1-102>/<filename>
    --arch: architecture to use for training (either 'vgg' or 'resnet')
    --learn_rate: learning rate for the model (e.g. 0.001, 0.03, etc.)
    --hidden_units: number of hidden units for the model based on the model:
        -vgg: between 25088 and 102
        -resnet: between 512 and 102
    --epochs: training epochs
    --checkpoint_filename: checkpoint filename.  Defaults to 
        'checkpoint.pth'
    --gpu: used to run the script using a GPU if available.
OUTPUTS:
    - checkpoint.pth file (or checkpoint_filename)

'''

## Imports
import argparse
import torch
import josh_flower_classifier as jfc


## Set Arguments
parser = argparse.ArgumentParser(description='arguments for train.py')

parser.add_argument('--image_directory', type = str, 
                    default = 'flowers',
                    help = 'prepared directory with train/test/valid flower images')
parser.add_argument('--arch', type = str, 
                    default = 'resnet',
                    help = 'model type, either "resnet" or "vgg"')
parser.add_argument('--learn_rate', type = float, 
                    default = 0.003,
                    help = 'learning rate')
parser.add_argument('--hidden_units', type = int, 
                    default = 256,
                    help = ('hidden layer nodes greater than 102, but less '
                            'than 25088 for vgg models and 512 for resnet '
                            'models.'))
parser.add_argument('--epochs', type = int, default = 4,
                    help = 'number of epochs for training, recommend 1-4')
parser.add_argument('--checkpoint_filename', type = str, 
                    default = 'checkpoint.pth',
                    help = 'filename with .pth extension')
parser.add_argument('--gpu', action="store_true",
                    help = ('Add this argument if you want to use a cuda '
                            'capable GPU to execute this script.'))


## Set variables to arguments
args = parser.parse_args()

image_directory = args.image_directory
arch = args.arch
learn_rate = args.learn_rate
hidden_units = args.hidden_units
epochs = args.epochs
checkpoint_filename = args.checkpoint_filename
gpu = args.gpu

# Handle whether the GPU will be used based on parameter provided.
if gpu and torch.cuda.is_available():
    device = torch.device('cuda')
    print('This script will be run using device [cuda]...')
elif gpu and torch.cuda.is_available() == False:
    device = torch.device('cpu')
    print('You selected --gpu, but your device is not cuda capable.  '
          'This script will be run using device [cpu]...')
elif gpu == False:
    device = torch.device('cpu')
    print('This script will be run using device [cpu]...')


## Create new model/optimizer/criterion based on inputs provided.
model, optimizer, criterion = jfc.create_flower_network(arch, learn_rate, 
                                                        hidden_units)

## prepare data loaders for training the model based on provided image_directory
a, b, c, d = jfc.prepare_dataloaders(image_directory)
train_loader = a
test_loader = b
valid_loader = c
class_to_idx = d

## Set the model Class_to_idx value
model.class_to_idx = class_to_idx

## Run an initial test of the created untrained model
jfc.testDataset('Test on Untrained Model', test_loader, model, criterion, 
                device)

## Train the model
model, optimizer = jfc.trainModel(model, train_loader, test_loader, optimizer, 
                                  criterion, epochs, device)

## Validate Training Model with validation data and save a checkpoint
jfc.modelCheckpoint(model, optimizer, criterion, valid_loader, 
                    checkpoint_filename, learn_rate, hidden_units, arch,
                    device)













