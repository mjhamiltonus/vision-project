from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.models import Sequential
from keras.callbacks import Callback
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import keras
import subprocess
import os

#MJH


from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

from keras.preprocessing import image


import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config

config.batch_size = 32
config.num_epochs = 20
IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
input_shape = (48, 48, 1)

FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172

def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model


def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
  Args:
    model: keras model
  """
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])



def load_fer2013():
    if not os.path.exists("fer2013"):
        print("Downloading the face emotion dataset...")
        subprocess.check_output("curl -SL https://www.dropbox.com/s/opuvvdv3uligypx/fer2013.tar | tar xz", shell=True)
    data = pd.read_csv("fer2013/fer2013.csv")
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = np.asarray(pixel_sequence.split(' '), dtype=np.uint8).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (width, height))
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()

    val_faces = faces[int(len(faces) * 0.8):]
    val_emotions = emotions[int(len(faces) * 0.8):]
    train_faces = faces[:int(len(faces) * 0.8)]
    train_emotions = emotions[:int(len(faces) * 0.8)]
    
    return train_faces, train_emotions, val_faces, val_emotions

# loading dataset

train_faces, train_emotions, val_faces, val_emotions = load_fer2013()

num_samples, num_classes = train_emotions.shape

train_faces /= 255.
val_faces /= 255.

#### My part - try building in inception and retrain last layer

# setup model
base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
model = add_new_last_layer(base_model, num_classes)

# fine-tuning
setup_to_finetune(model)

# preprocess - apply in the image generator to ge tto Inception input size IM_HEIGTH
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    rescale=IM_HEIGHT/input_shape[0]
)

# retrain
# MJH Get this to work here!
train_faces = np.repeat(train_faces[:, :, :, 0], 3, axis=3)
print(" train-faces: {:}   train-emotions: {:}".format(train_faces.shape, train_emotions.shape))
# replicate fake RGB?




model.fit_generator(datagen.flow(train_faces, train_emotions, batch_size=config.batch_size),
        epochs=config.num_epochs, steps_per_epoch=len(train_faces)/config.batch_size, verbose=1, callbacks=[
            WandbCallback(data_type="image", labels=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
        ], validation_data=datagen.flow(val_faces, val_emotions))
#model.fit_generator(train_faces, train_emotions, batch_size=config.batch_size,
#        epochs=config.num_epochs, verbose=1, callbacks=[
#            WandbCallback(data_type="image", labels=["Angry", "Disgust", "Fear", "Happy", #"Sad", "Surprise", "Neutral"])
#        ], validation_data=(val_faces, val_emotions))



## end modification
# this is called in setup to finetune
#model.compile(optimizer='adam', loss='categorical_crossentropy',
#metrics=['accuracy'])



model.save("emotion.h5")



