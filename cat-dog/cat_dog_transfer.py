

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


splits = tfds.Split.ALL.subsplit(weighted=(80, 20))

splits, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True, split = splits)

(train_examples, validation_examples) = splits

def format_image(image, label):
  # `hub` image modules exepct their data normalized to the [0,1] range.
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return  image, label

num_examples = info.splits['train'].num_examples

BATCH_SIZE = 32
IMAGE_RES = 224

train_batches      = train_examples.cache().shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).repeat().prefetch(1)
validation_batches = validation_examples.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)


URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(IMAGE_RES, IMAGE_RES,3))

feature_extractor.trainable = False

model = tf.keras.Sequential([
  feature_extractor,
  layers.Dense(2, activation='softmax')
])

model.summary()


model.compile(
  optimizer='adam', 
  loss=tf.losses.SparseCategoricalCrossentropy(),
  metrics=['accuracy'])

dog = Image.open("./dog.jpeg")
dog  = tf.keras.preprocessing.image.img_to_array(dog )
dog = tf.image.resize(dog, (IMAGE_RES, IMAGE_RES))/255.0

cat = Image.open("./cat.jpeg")
cat  = tf.keras.preprocessing.image.img_to_array(cat )
cat = tf.image.resize(cat, (IMAGE_RES, IMAGE_RES))/255.0

EPOCHS = 3
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches,
                    steps_per_epoch=500)



dog_result = model.predict(dog[np.newaxis, ...])
cat_result = model.predict(cat[np.newaxis, ...])

print("DOG: ", dog_result)
print("CAT: ", cat_result)



class_names = np.array(info.features['label'].names)
class_names

image_batch, label_batch = next(iter(validation_batches.take(1)))

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()
predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]


print("Labels: ", label_batch)
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