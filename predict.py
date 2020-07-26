#Import Python Modules
import numpy as np
import matplotlib.pyplot as plt
import PIL
import argparse
import json
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets,transforms, models
from collections import OrderedDict
from PIL import Image
from time import time, sleep

#Import functions created for this program
from process_image import process_image

def main():
    start_time = time()
    
    def get_input_args():
        """
        Retrieves and parses the command line arguments provided by the user when
        they run the program from a terminal window. This function uses Python's 
        argparse module to created and defined these command line arguments. If 
        the user fails to provide some or all of the arguments, then the default 
        values are used for the missing arguments. 
  
        """
        # Create Parse using ArgumentParser

        parser = argparse.ArgumentParser()

        # Create command line arguments as mentioned above using add_argument() from ArguementParser method

        parser.add_argument('--image_path',type=str,default='flowers/test/10/image_07090.jpg',help='path for image to predict')
        parser.add_argument('--save_dir',type=str,default='fc_checkpoint.pth',help='path for checkpoint')
        parser.add_argument('--topk',type=int,default=5,help='input number of top classes for prediction')
        parser.add_argument('--arch', type = str, default = 'vgg16', help = 'architecure of Model') 
        parser.add_argument('--gpu',default=True,help='use GPU to make predictions')
        parser.add_argument('--cat_to_name', default = 'cat_to_name.json',help='enters a path to image.')

        in_arg = parser.parse_args()

        return in_arg
    
    in_arg = get_input_args()
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    squeezenet = models.squeezenet1_0(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    densenet = models.densenet161(pretrained=True)
    inception = models.inception_v3(pretrained=True)

    models_dict = {'resnet':resnet18
                   ,'alexnet':alexnet
                   ,'squeezenet':squeezenet
                   ,'vgg16':vgg16
                   ,'densenet':densenet
                   ,'inception':inception
                  }
    
    def load_checkpoint(path):
        checkpoint = torch.load(path)
        if checkpoint['arch'] == in_arg.arch:
            model = models_dict[in_arg.arch]

            for param in model.parameters():
                param.requires_grad = False

            model.class_to_idx = checkpoint['class_to_idx']
            model.classifier = checkpoint['classifier']
            model.load_state_dict(checkpoint['model_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epochs = checkpoint['epochs']

        return model
    
    trained_model = load_checkpoint(in_arg.save_dir)
    print('Checkpoint loaded')

    def predict(image_path, model, topk):
        
        ''' Predict the class of an image using a trained model.
        '''
        #Preprocess the image
        im = process_image(image_path)
        print('Image preprocessed')
        
        im = im.unsqueeze_(0)
        im = im.float()
        
        print('image type is: {}'.format(type(im)))
        print('Beginning Model Eval')

        if torch.cuda.is_available() and in_arg.gpu == True:
            model.cuda()
        #Pass through the model
        model.eval()
        with torch.no_grad():
            logps = model.forward(im.cuda())

        ps = torch.exp(logps)
        
        #Find top-k probabilities and indices 
        top_probability, indices = torch.topk(ps, dim=1, k=topk)
        
        #Find the class using the indices
        indices = np.array(indices) 
        index_to_class = {val: key for key, val in model.class_to_idx.items()} 
        top_classes = [index_to_class[i] for i in indices[0]]

        #Map the class name with collected top-k classes
        names = []
        for classes in top_classes:
                names.append(cat_to_name[str(classes)])
                
        print('prediction complete')
        return top_probability.cpu().numpy(), names
    
    top_probability = predict(in_arg.image_path,trained_model,in_arg.topk)[0]
    top_classes = predict(in_arg.image_path,trained_model,in_arg.topk)[1]
    print(top_probability,top_classes)
    
    end_time = time()
    
    tot_time = end_time - start_time
    
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

if __name__=="__main__":
        main()
    