import pandas as pd
import cv2
import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--dataset_path', required=True, help='path to dataset')
args = parser.parse_args()

path = args.dataset_path

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprised", "neutral"]
emotion_list = list()
pixels_list = list()

for (root, subdirs, file_names) in os.walk(path):
    if (len(file_names) != 0):
        for file_name in file_names:
            pixels = cv2.imread(root + '/' +  file_name, 0)
            if pixels == None:
                continue
            print(pixels)
            pixels_list.append(pixels)
            
            emotion = file_name[:file_name.find('.')].lower()
            emotion_index = -1
            for std_emotion_index in range(len(EMOTIONS)):
                if (EMOTIONS[std_emotion_index] in emotion):
                    emotion_index = std_emotion_index
            if emotion_index == -1:
                print('ERROR::emotion index is -1')
            emotion_list.append(emotion_index)
df = pd.DataFrame({"Emotion": emotion_list, "pixels": pixels_list})
print(df)