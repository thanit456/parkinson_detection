import re

def change_main_directory( full_path, start_dir, dest_dir ):
    ''' This function for changing the main directory from start_dir to dest_dir
    '''
    return re.sub( start_dir, dest_dir, full_path)
