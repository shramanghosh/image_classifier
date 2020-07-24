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
from get_input_args import get_input_args
from load import get_data
from checkpoint import load_checkpoint

def main():
    start_time = time() #Time the program
    
    in_arg = get_input_args()
    
    #Retrieve data    
    train_data = get_data(in_arg.dir)[0]
    trainloader = get_data(in_arg.dir)[1]
    validloader = get_data(in_arg.dir)[2]
    testloader = get_data(in_arg.dir)[3]
    
    #Obatain labels
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    #Train model
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
    
    def training_model(model_name, input_units, hidden_units, output_units,epoch_number,learn_rate):
        """
        Trains a neural network using a pretrained model by:
        Defining a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
        Training the classifier layers using backpropagation using the pre-trained network to get the features
        Tracking the loss and accuracy on the validation set to determine the best hyperparameters

        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = models_dict[model_name]

        for param in model.parameters():
            param.requires_grad=False

        model.classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(input_units, hidden_units, bias=True)),
                                  ('relu1', nn.ReLU()),
                                  ('dropout1', nn.Dropout(p=0.5)),
                                  ('fc2', nn.Linear(hidden_units, output_units, bias=True)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))

        print("classifier updated")
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(),lr=learn_rate)

        model.to(device);

        epochs = epoch_number
        steps = 0
        running_losses = 0
        testing_losses = []
        training_losses = []
        print_every = 10 #how many steps we're going to go before we print validation loss

        print("assignments complete, commencing training")

        for e in range(epochs):
            model.train()
            for inputs,labels in trainloader: 
                steps += 1 
                inputs,labels = inputs.to(device),labels.to(device)

                optimizer.zero_grad()
                logps = model.forward(inputs)
                loss = criterion(logps,labels)
                loss.backward()
                optimizer.step()

                running_losses += loss.item()

                if steps % print_every == 0:#every 5 steps we drop out of training loop to test accuracy
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs,labels in validloader:
                            inputs,labels = inputs.to(device),labels.to(device)

                            logps = model.forward(inputs)
                            batch_loss = criterion(logps,labels)

                            test_loss += batch_loss.item()

                            ps = torch.exp(logps)
                            top_p,top_class = ps.topk(1,dim=1)
                            equals = top_class == labels.view(*top_class.shape)

                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {e+1}/{epochs}.. " #Keep track of where we are
                          f"Train loss: {running_losses/print_every: .3f}.. " #Average of trainingloss
                          f"Test loss: {test_loss/len(validloader): .3f}.. " 
                          #Denominator: How many batches are in valid dataset
                          #Numerator: Sum of test losses across batches
                          #Result: average test loss
                          f"Test accuracy: {accuracy/len(validloader): .3f}") #Average accuracy

                    running_losses = 0
                    model.train()
                    
        model.class_to_idx = train_data.class_to_idx
        print("Training Complete")
        
        print("Testing the model")

        #Test model
        testset_loss = 0
        test_accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs,labels in testloader:
                inputs,labels = inputs.to(device),labels.to(device)

                logps = model.forward(inputs)
                testset_loss += criterion(logps,labels).item()

                ps = torch.exp(logps)
                top_p,top_class = ps.topk(1,dim=1)
                equals = top_class == labels.view(*top_class.shape)

                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        model.class_to_idx = train_data.class_to_idx
        print(f"Epoch {e+1}/{epochs}.. " #Keep track of where we are
              f"Test loss: {testset_loss/len(testloader): .3f}.. " 
                          #Denominator: How many batches are in valid dataset
                          #Numerator: Sum of test losses across batches
                          #Result: average test loss
              f"Test accuracy: {test_accuracy/len(testloader): .3f}") #Average accuracy

        return model, model.state_dict(), model.classifier, model.class_to_idx
    
    trained_model = training_model(in_arg.arch,in_arg.input_units,in_arg.hidden_units,in_arg.output_units,
                           in_arg.epoch_number,in_arg.learn_rate)
    #Save Checkpoint
    path = in_arg.save_dir

    torch.save({'arch':in_arg.arch,
                'input':in_arg.input_units,
                'output':in_arg.output_units,
                'epochs':in_arg.epoch_number,
                'model_state_dict':trained_model[1],
                #'optimizer_state_dict':optimizer.state_dict(),
                'classifier':trained_model[2],
                'class_to_idx':trained_model[3]
                },path)
    
    print("Checkpoint saved")
   
    end_time = time()
    
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    
if __name__=="__main__":
        main()