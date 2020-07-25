# image_classifier

image_classifier is a deep learning model that leverages pretrained neural networks in order to classify images of different species of flowers.

This was developed for the AI Programming with Python Nanodegree offered by Udacity - https://www.udacity.com/course/ai-programming-python-nanodegree--nd089

# Format 

The image_classifier consists of two separate projects both of which however have the structure and purpose. 

The starting point of this repository is the Image Classifier Project.ipynb which is a Jupyter notebook that provides ind-depth instructions on how to build
this project in a step-by-step manner from training the network, testing it and leveraging it to make predictions as a sanity check.

The rest of the files follow the same structure but are split across multiple python files to be used as an application directly from the command line. 

# Structure

#### Training ####

The first step is to train the network, which is accomplished in train.py. 

Prior to using train.py ensure you have both get_input_args.py and load.py in the same directory as well as cat_to_name.json which contains the labels. 

Train.py merely trains your network using the pretrained deep learning model you wish to use (the default being vgg16) and saves it as a checkpoint. 

Assuming the code runs fine you should be able to attain an accuracy over 70% 

#### Prediction ####

Next, you're ready to run a prediction which is accomplished by predict.py. 

Start by ensuring process_image.py is in the same directory as well as the path to the checkpoint you saved at the end of train.py. 

