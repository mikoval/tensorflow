import tensorflow as tf

import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#input data
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

#print data
#for i,c in enumerate(celsius_q):
#    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

#1 input, 1 output for layer
#l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

#model takes in the layer defined above
#model = tf.keras.Sequential([l0])

#more layers
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([l0, l1, l2])


#compile the model
#Loss function - A way of measuring how far off predictions are from the desired outcome. (The measured difference is called the "loss".)
#Optimizer function - A way of adjusting internal values in order to reduce the loss.
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

#train
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

#optionally show the results
#import matplotlib.pyplot as plt
#plt.xlabel('Epoch Number')
#plt.ylabel("Loss Magnitude")
#plt.plot(history.history['loss'])
#plt.show(block=True)

#value = 100.0
#print("Estimating : {} -> {}".format(value,  model.predict([value])[0][0]))

while True:
    value = eval(raw_input("Enter a number: "))
    print("Estimating : {} -> {}".format(value,  model.predict([value])[0][0]))

