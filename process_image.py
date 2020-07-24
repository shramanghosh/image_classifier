from PIL import Image
import numpy as np
import torch
import argparse
from get_input_args import get_input_args
from load import get_data

def process_image(image):
    ''' Steps:
    
    1. Resize an image
    2. Center Crop
    3. Convert to numpy array
    4. Make color channel dimension first instead of last
    5. Normalize
    6. Convert to tensor
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    img = Image.open(image)
    print(type(img))
    
    width, height = img.size
    print('Original dimensions: {}'.format(img.size))
    
    img = img.resize((256, int(256*(height/width))) if width < height else (int(256*(width/height)), 256))
    
    #Get the dimensions of the new image size
    width, height = img.size
    print('New dimensions: {}'.format(img.size))
    
    #Center Crop
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    
    img = np.array(img)
    print(type(img))
    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))
    
    # Make all values between 0 and 1
    img = img/256
    
    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    
    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    print(type(image))
    
    return image