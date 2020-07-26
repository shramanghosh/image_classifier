# Imports python modules
import argparse

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
    
    parser.add_argument('--data_dir', type = str, default = 'flowers', 
                    help = 'path to the folder of flower images')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'architecure of Model')
    parser.add_argument('--hidden_units',type=int,default=4096,help='hidden units for model selected')
    parser.add_argument('--epoch_number',type=int,default=3,help='number of epochs')
    parser.add_argument('--learn_rate',type=int,default=0.001,help='input learn rate for optimizer')
    parser.add_argument('--save_dir',type=str,default='fc_checkpoint.pth',help='path for checkpoint')
    parser.add_argument('--gpu',default=True,help='use GPU to train the model')
    
    in_arg = parser.parse_args()
    
    return in_arg