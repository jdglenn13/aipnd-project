# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

This project has leveraged the pytorch pre-trained models Resnet34 and VGG13.  The VGG13 models will result in a larger checkpoitn file due to the larger number of layers/hidden units within the network, which also resulted in a more lengthy time to train.  Based on my testing, I was able to achieve higher accuracy with the Resnet models than the VGG models.


# Installation
**NOTE:** The josh_flower_classifier.py, train.py, and predict.py scripts were created on a Conda environment running Python 3.11 with the packages defined in spec-file.txt.  The Udacity training environment won't let me set up a conda environment running Python 3.11 and the related packages, so I'm unable to test my scripts as originally written and tested in the udacity environment.  However, I have created equivalent scripts with `_udacity` in the filename that should work in the udacity workspaces.  Code has been changed in the scripts to accomodate the differences in versions in the udacity workspace and the workspace_utils.py script is also required to maintain the session in the udacity workspace for training.

The following files within the ImageClassifier folder are required for use with this progam:
* josh_flower_classifier.py
* train.py
* predict.py
* cat_to_name.json

The default arguments of train.py make use of the already loaded 'flowers' directory.

# Commands for Testing scripts:
The following command will train the model using the default values defined in train.py (image_directory=flowers, arch=resnet, learn_rate=0.003, hidden_units=256, epochs=4):

`python train.py --checkpoint_file checkpoint_resnet.pth`

The following command will train the model using the base VGG13 model.  Note the hidden units argument is used due to the VGG network's large number of starting nodes in the classifier to split the difference a bit more:

`python train.py --checkpoint_file checkpoint_vgg.pth --arch vgg --hidden_units 1024`

The following commands will test the network on two flower images taken directly from the validation test set.  The predict.py script has the ability to plot the results in matplotlib with '--visualize', but that is excluded here so the results are just printed to the screen:

`python predict.py --image_file inference/14_image_06082.jpg --image_class 14 --checkpoint_file checkpoint_resnet.pth --topk 5`

`python predict.py --image_file inference/82_image_01659.jpg --image_class 82 --checkpoint_file checkpoint_resnet.pth --topk 5`


