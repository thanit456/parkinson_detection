from imutils import face_utils
from imutils.face_utils import FACIAL_LANDMARKS_IDXS
import imutils
import numpy as np 
import pandas as pd
import argparse
import dlib
import cv2

# helper function

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
parser.add_argument('-i', '--image', required=True, help='path to input image')
args = vars(parser.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

image = cv2.imread(args['image'])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    
    clone = image.copy()
    cv2.putText(clone, body_part, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 0, 255), 2)

    body_part_shape = shape[FACIAL_LANDMARKS_IDXS[body_part][0]: FACIAL_LANDMARKS_IDXS[body_part][1]]

    for (x, y) in body_part_shape:
        cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

    (x, y, w, h) = cv2.boundingRect(np.array(body_part_shape))
    roi = clone[y: y+h, x: x+w]
    roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
    
    cv2.imshow('ROI', roi)
    # cv2.imshow('Image', clone)
    cv2.waitKey(0)
