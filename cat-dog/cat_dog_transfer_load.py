

from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import os
import numpy as np
import time

import urllib
import cv2
from tensorflow.keras import layers

import PIL.Image as Image


# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image

IMAGE_RES = 224

reloaded = tf.keras.models.load_model(
  "./keras.h5", 
  # `custom_objects` tells keras how to load a `hub.KerasLayer`
  custom_objects={'KerasLayer': hub.KerasLayer})

reloaded.summary()



dog = Image.open("./dog.jpeg")
dog  = tf.keras.preprocessing.image.img_to_array(dog )
dog = tf.image.resize(dog, (IMAGE_RES, IMAGE_RES))/255.0

cat = Image.open("./cat.jpeg")
cat  = tf.keras.preprocessing.image.img_to_array(cat )
cat = tf.image.resize(cat, (IMAGE_RES, IMAGE_RES))/255.0


dog_result = reloaded.predict(dog[np.newaxis, ...])
cat_result = reloaded.predict(cat[np.newaxis, ...])

print("DOG: ", np.argmax(dog_result, axis=-1))
print("CAT: ", np.argmax(cat_result, axis=-1))






