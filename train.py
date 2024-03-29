import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Convolution2D, Dropout, Conv2D
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, MaxPooling2D, SeparableConv2D, Flatten

from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, Model

from sklearn.model_selection import train_test_split

import pandas as pd
import cv2
import numpy as np
import os
import argparse

# Global variables
batch_size = 32
num_epochs = 110
input_shape = (48, 48, 1)
verbose = 1
num_classes = 7
patience = 50
base_path = 'model/'
l2_regularization = 0.01
dataset_path = './fer2013/fer2013.csv'
EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprised", "neutral"]
IMAGE_EXTENSIONS = [".jpeg", ".jpg", ".png"]

# argparser
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_path', required=True, help='path to dataset')
parser.add_argument('-c', '--csv_path', default='./fer2013/fer2013.csv', help='path to save .csv file')
args = vars(parser.parse_args())

# helper function
def read_data2csv( dir_path, csv_path ):
    
    columns_df = ['emotion', 'pixels']
    
    df = pd.DataFrame(list(), columns=columns_df)
    for (root, subdirs, file_names) in os.walk(dir_path):
        if (len(file_names) != 0):
            for file_name in file_names:

                # preprocess file name
                # extract image extension out of file name and get emotion that corresponds to the image

                order_emotion = -1

                for image_extension in IMAGE_EXTENSIONS:
                    if (image_extension in file_name):
                        preprocessed_file_name = (file_name.replace(image_extension, '')).strip().lower()

                for index_emotions in range(len(EMOTIONS)):
                    # contempt ?

                    if (EMOTIONS[index_emotions] in preprocessed_file_name):
                        order_emotion = index_emotions
                        break    
                else:
                    print('ERROR::NO EMOTION ACCORDING TO MY TRAINING')

                if (order_emotion != -1):
                    print(root+'/'+file_name)
                    image = cv2.imread(root + '/' + file_name, 0)
                    image = cv2.resize(image, (48, 48))
                    lin_image = image.flatten()
                    pixel_list = lin_image.tolist()
                    pixel_str_list = map(str, pixel_list)
                    pixel_str = ' '.join(pixel_str_list)
                    print('Finished generate image pixel string')

                    sub_df = pd.DataFrame([[order_emotion, pixel_str]], columns=columns_df)
                    #print('EMOTION :', type(order_emotion))
                    #print('pixels str: ', type(pixel_str))
                    print(sub_df)
                    df.append(sub_df)
                    #print(df)
                    print('Append to the main df')          

    df.to_csv(args['csv_path'], index=None, header=True) 
    print('Save finished df all faces')

def load_data_from_csv( dir_path ):
    data = pd.read_csv(dir_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (48, 48))
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    return faces, emotions


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


# Data generator
data_generator = ImageDataGenerator( featurewise_center=False,
                                    featurewise_std_normalization=False,
                                    rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=.1,
                                    horizontal_flip=True )

# model parameters
regularization = l2(l2_regularization)

# base
img_input = Input(input_shape)
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
 
# module 1
residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])
 
# module 2
residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])
 
# module 3
residual = Conv2D(64, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(64, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(64, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])
 


# module 4
residual = Conv2D(128, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(128, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(128, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])
x = Conv2D(num_classes, (3, 3), padding='same')(x)
x = GlobalAveragePooling2D()(x)
output = Activation('softmax',name='predictions')(x)
 
model = Model(img_input, output)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
 
# callbacks
log_file_path = base_path + '_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1)
trained_models_path = base_path + '_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]
 

#read_data2csv(args['dataset_path'], args['csv_path'])

#loading dataset 
faces, emotions = load_data_from_csv( args['dataset_path'] )
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)
model.fit_generator(data_generator.flow(xtrain, ytrain,batch_size),
                        steps_per_epoch=len(xtrain) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=(xtest,ytest))
