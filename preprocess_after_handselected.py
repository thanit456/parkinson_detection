import os 
import argparse
import re

# import from local
import utils


# helper function 
def simple_file_name_correction( file_name ):
    ''' This function for correcting filename ( fit for parkinson dataset ) to be the simplest form
    '''
    

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True, help='path to walk through everything in the directory')
args = parser.parse_args()

input_dir = args.input_dir

for (root, subdirs, file_names) in os.walk(input_dir):
    if (len(file_names) != 0):
        for file_name in file_names:
            