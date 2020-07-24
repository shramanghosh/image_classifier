# Imports python modules
import argparse

def get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder as --dir with default value 'pet_images'
      2. CNN Model Architecture as --arch with default value 'vgg'
      3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    
    parser = argparse.ArgumentParser()
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    
    parser.add_argument('--dir', type = str, default = 'flowers', 
                    help = 'path to the folder of flower images')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'architecure of Model')
    parser.add_argument('--input_units', type=int, default=25088,help='input units for model selected')
    parser.add_argument('--hidden_units',type=int,default=4096,help='hidden units for model selected')
    parser.add_argument('--output_units',type=int,default=102,help='number of output classes')
    parser.add_argument('--epoch_number',type=int,default=3,help='number of epochs')
    parser.add_argument('--learn_rate',type=int,default=0.001,help='input learn rate for optimizer')
    parser.add_argument('--save_dir',type=str,default='fc_checkpoint.pth',help='path for checkpoint')
    parser.add_argument('--image_path',type=str,default='flowers/test/10/image_07090.jpg',help='path for image to predict')
    parser.add_argument('--topk',type=int, default=5,help='input number of top classes for prediction')
    
    in_arg = parser.parse_args()
    
    return in_arg