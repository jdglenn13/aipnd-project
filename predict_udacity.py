'''
filename: predict.py
description: will use a pre-trained model created using train.py to predict
    the type of flower in the image and 
author: Joshua Glenn (jglenn@its.jnj.com)
last modified: 11Apr2024

INPUTS: Arguments to the predict.py script
    --image_file : str
    --image_class : str
    --checkpoint_file : str
    --topk : int
    --visualize : bool
    
OUTPUTS: Displays image with graph showing topk classes.
'''

## Imports
import argparse
import josh_flower_classifier_udacity as jfc


## Set Arguments
parser = argparse.ArgumentParser(description='arguments for predict.py')

parser.add_argument('--image_file', type = str,
                    help = ('REQUIRED - image file with path to be assessed '
                            'tp preduct the type of flower.'),
                    required = True)
parser.add_argument('--image_class', type = str, 
                    default = '0',
                    help = ('1-102 key for the flower type in ' 
                            'cat_to_name.json. defaults to 0 ' 
                            "if you don't know"))
parser.add_argument('--checkpoint_file', type = str, 
                    help = ('REQUIRED - .pth file created from the train.py '
                            'script.'),
                    required = True)
parser.add_argument('--topk', type = int, 
                    help = 'topk classes to display in the output, 1-5',
                    default = 5)
parser.add_argument('--visualize', action="store_true", 
                    help = 'Add this argument to visualize output in matplotlib')



## Set variables to arguments
args = parser.parse_args()

image_file = args.image_file
image_class = args.image_class
checkpoint_file = args.checkpoint_file
topk = args.topk
visualize = args.visualize

## Load model from specified checkpoint file
model, optimizer, criterion = jfc.load_checkpoint(checkpoint_file)

## predict the classification of the provided image
jfc.predict(image_file, image_class, model, topk, visualize)
