import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import imutils
import cv2
import numpy as np
import sys
import argparse
import os

# import from local
import utils

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--haar_face', default='./haarcascade_frontalface_default.xml', help='xml file for haar cascade frontal face detection')
parser.add_argument('--pretrained_weights', default='./model/pretrained/_mini_XCEPTION.106-0.65.hdf5', help='pretrained weights for each model')
parser.add_argument('--image_dir', required=True, help='path to input image')
parser.add_argument('--save_dir', required=True, help='directory path to save test images')
args = parser.parse_args()

# parameters for loading data and images
detection_model_path = args.haar_face
emotion_model_path = args.pretrained_weights
image_dir = args.image_dir
save_dir = args.save_dir

# hyper parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprised", "neutral"]


for (root, subdirs, file_names) in os.walk(image_dir):
    
    # create directory according to the directory system
    for subdir in subdirs:
        save_root = change_main_directory(root, image_dir, save_dir)
        if (not os.path.exists(save_root + '/' + subdir)):
            os.makedirs(save_root + '/' + subdir)
    
    if (len(file_names) != 0): 
        for file_name in file_names:

            # Determine image path
            img_path = root + '/' + file_name
            save_img_path = utils.change_main_directory(img_path, image_dir, save_dir)

            # reading the frame
            orig_frame = cv2.imread(img_path)
            frame = cv2.imread(img_path, 0)
            faces = face_detection.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            if len(faces) != 0:
                faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = faces
                roi = frame[fY: fY + fH, fX: fX + fW]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                preds = emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]
                cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
                
                cv2.imwrite(save_img_path, orig_frame)
                print('Save finished : ' + save_img_path)


                