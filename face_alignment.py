# import packages
from imutils.face_utils import FaceAligner

# for HOG-based
# from imutils.face_utils import rect_to_bb

# import from local
import utils

import argparse
import imutils
import dlib
import cv2
import os
import re

# global variable
image_extension = ['.jpg', '.jpeg', '.png']

# helper function 
def save_aligned_face( aligned_face, root, raw_dir, save_dir, file_name, bb_index ):
    ''' This function to save an aligned face according to the directory
    '''
    # manipulate aligned face path to save 
    aligned_save_path = root + '/' + file_name[ : file_name.rfind('.') ]
    aligned_save_path = utils.change_main_directory( aligned_save_path, raw_dir, save_dir )
    if ( bb_index != 0 ):
        aligned_save_path += '_' + str( bb_index )
    aligned_save_path += file_name[ file_name.rfind('.') :  ]

    # save aligned face into the directory
    cv2.imwrite( aligned_save_path, aligned_face )
    
    print('Save finished : ', aligned_save_path )

def face2bb( rect ):
    ''' This function to transform a face to a bounding box
    '''
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y

    return (x, y, w, h)

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cnn_weight', default="mmod_human_face_detector.dat", help="path to pretraineds face detector")
parser.add_argument('--shape_detector', default="shape_predictor_68_face_landmarks.dat", help="path to pretrained shape detector")
parser.add_argument('--raw_dir', required=True, help="path to raw faces directory")
parser.add_argument('--save_dir', required=True, help="path to save aligned faces")
args = parser.parse_args()

# dlib's face detector (HOG-based)
# detector = dlib.get_frontal_face_detector()

# dlib's face detector (CNN-based)
detector = dlib.cnn_face_detection_model_v1(args.cnn_weight)
predictor = dlib.shape_predictor(args.shape_detector)

# create face aligner
fa = FaceAligner(predictor, desiredFaceWidth=256)


print('Start to walk through raw data directory')
for (root, subdirs, file_names) in os.walk(args.raw_dir):
   
    # create directory according to the directory system
    for subdir in subdirs:
        save_root = utils.change_main_directory( root, args.raw_dir, args.save_dir )
        if ( not os.path.exists( save_root + '/' + subdir) ):
            os.makedirs( save_root + '/' + subdir )
   
    # read the face image
    if (len(file_names) != 0):

        # print('##########')
        # print(root, end=', ')
        # print(*subdirs, end=', ')
        # print(*file_names)

        # create save root path
        rootSaveDir = args.save_dir + root[ root.find('/') : ]

        for file_name in file_names:
            
            if file_name[file_name.rfind('.'):] in image_extension:   
                image = cv2.imread(root + '/' + file_name)
                image = imutils.resize(image, width=800)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # detect the face
                faces = detector(gray, 2)

                # initialize the index of rectangles
                bb_index = 0

                # loop over the face detections
                for face in faces:
                    # extract ROI of the original face, then align the face
                    # using facial landmark
                    
                    # for HOG-based  
                    # (x, y, w, h) = rect_to_bb(rect)

                    # for CNN-based
                    (x, y, w, h) = face2bb( face ) 

                    #?
                    aligned_face = fa.align( image, gray, face.rect )

                    # save the aligned face image at save_dir directory
                    save_aligned_face( aligned_face, root, args.raw_dir, args.save_dir, file_name, bb_index) 

                    # increment index of rectangles 
                    bb_index += 1
