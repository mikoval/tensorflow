

from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import os
import numpy as np
import time

import PIL.Image as Image
from tensorflow.keras import layers

avail = tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)

print("GPU AVAIL: " , avail)


from tensorflow.keras.preprocessing.image import ImageDataGenerator


splits = tfds.Split.TRAIN.subsplit([70, 30])

(training_set, validation_set), dataset_info = tfds.load('tf_flowers', with_info=True, as_supervised=True, split=splits)

num_classes = dataset_info.features['label'].num_classes

num_training_examples = 0
num_validation_examples = 0

for example in training_set:
  num_training_examples += 1

for example in validation_set:
  num_validation_examples += 1

print('Total Number of Classes: {}'.format(num_classes))
print('Total Number of Training Images: {}'.format(num_training_examples))
print('Total Number of Validation Images: {} \n'.format(num_validation_examples))


BATCH_SIZE = 32
IMAGE_RES = 224


def format_image(image, label):
  # `hub` image modules exepct their data normalized to the [0,1] range.
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return  image, label



train_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(BATCH_SIZE).repeat().prefetch(1)

validation_batches = validation_set.map(format_image).batch(BATCH_SIZE).repeat().prefetch(1)


URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(IMAGE_RES, IMAGE_RES, 3))

feature_extractor.trainable = False

model = tf.keras.Sequential([
  feature_extractor,
  layers.Dense(num_classes, activation='softmax')
])

model.summary()


model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

EPOCHS = 6

history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches,
                    steps_per_epoch=500,
                    validation_steps=500)



class_names = np.array(dataset_info.features['label'].names)

print(class_names)

image_batch, label_batch = next(iter(train_batches))


image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()

predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]

print(predicted_class_names)

print("Labels:           ", label_batch)
print("Predicted labels: ", predicted_ids)

#export keras model
t = time.time()

export_path_keras = "keras.h5".format(int(t))
print("KERAS NAME : ", export_path_keras)

model.save(export_path_keras)

#export SavedModel
t = time.time()

export_path_sm = "./{}".format(int(t))
print("SAVED MODEL NAME: " , export_path_sm)

tf.saved_model.save(model, export_path_sm)