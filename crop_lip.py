from imutils import face_utils
from imutils.face_utils import FACIAL_LANDMARKS_IDXS
import imutils
import numpy as np 
import argparse
import dlib
import cv2
import pandas as pd
import os

# import local
import utils

# FACIAL_LANDMARKS_IDXS = OrderedDict([
# 	("mouth", (48, 68)),
# 	("right_eyebrow", (17, 22)),
# 	("left_eyebrow", (22, 27)),
# 	("right_eye", (36, 42)),
# 	("left_eye", (42, 48)),
# 	("nose", (27, 35)),
# 	("jaw", (0, 17))
# ])

body_part = 'mouth'

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--shape_predictor', default='./shape_predictor_68_face_landmarks.dat', help='pretrained weights for shape detection')
parser.add_argument('-i', '--input_dir', required=True, help='path to input image directory')
parser.add_argument('-s', '--save_dir', require=True, help='path to save directory')
args = vars(parser.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

for (root, subdirs, file_names) in os.walk(args['input_dir']):
    for subdir in subdirs:
        save_root = change_main_directory(root, args['input_dir'], args['save_dir'])
        if (not os.path.exists(save_root + '/' + subdir)):
            os.makedirs(save_root + '/' + subdir)

    if (len(file_names) != 0):
        for file_name in file_names:
            image = cv2.imread(root + '/' + file_name)
            image = imutils.resize(image, width=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            rects = detector(gray, 1)

            for (i, rect) in enumerate(rects):
                
                #####################
                
                 ### 1. CROP LIP ###

                #####################

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                
                # clone = image.copy()
                # #cv2.putText(clone, body_part, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7, (0, 0, 255), 2)

                body_part_shape = shape[FACIAL_LANDMARKS_IDXS[body_part][0]: FACIAL_LANDMARKS_IDXS[body_part][1]]

                # for (x, y) in body_part_shape:
                #     cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

                (x, y, w, h) = cv2.boundingRect(np.array(body_part_shape))
                roi = image[y: y+h, x: x+w]
                roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
                
                ##################################
                
                 ### 2.GATHER FEATURE POSITION ###

                ##################################
                
                shape = predictor
                
                
