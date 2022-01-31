# *****************************************************
#
# Project : Reconnaissance d'image Qualité Béton
# Project 6
# Auteur : Ali Naama
# Forked from :
# https://github.com/trushraut18/Image-Classifcation
#
# *****************************************************


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Flatten,Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import os, sys

# Description des classes du modèle - Qualité RMX
CLASS_NAMES = ['shade','efflorescence','ressuage','Black','pommeling','Rust','dustiness','stone','milt','bubbling','cracking','flaking']

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'DatasetP6\train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 classes  = CLASS_NAMES ,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(r'DatasetP6\test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            classes  = CLASS_NAMES ,
                                            class_mode = 'categorical')

# Adaptation des paramètres : Nombre de classes 12  v1

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(4, (3,3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(8, (2,2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
     tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(12, activation='softmax')
])
model.summary()

opt = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy',
                            optimizer=opt,
                            metrics=['accuracy'])


r = model.fit_generator(training_set,   steps_per_epoch = 60,
                                    epochs = 20,
                                    validation_steps = 300)

print('training finished!!')
print('saving weights')
# saisir le chemin où seront stockés les poids et modèle du modèle
# qui vient d'être entraîné
model.save_weights('D:/Projet Perso/Ali/Data Scientist/Projet 6/model/p6.hdf5')
model.save('D:/Projet Perso/Ali/Data Scientist/Projet 6/model/p6.model')
print('all weights saved successfully !!')

# Evaluate
test_loss, test_accuracy = model.evaluate(test_set, verbose=1)
print("Loss  : ", test_loss)
print("Accuracy  :",test_accuracy)
epoch = 120
epoch_range = range(1, epoch + 1)


# Test unitaire
test_image = image.load_img(r'DatasetP6\Test\black\Tachesnoires4tst.jpg', target_size = (64,64))
test_image = np.expand_dims(test_image, axis=0)

result = model.predict(test_image)
# vérification des résultats
print(result)


# Test unitaire
test_image2 = image.load_img(r'DatasetP6\Test\rust\rouille1.jpg', target_size = (64,64))
test_image2 = np.expand_dims(test_image2, axis=0)

result2 = model.predict(test_image2)
# vérification des résultats
print(result2)

input("Fin des traitements")
sys.exit()
