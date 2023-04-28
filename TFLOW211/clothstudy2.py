import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



train_images = train_images / 255.0
test_images = test_images / 255.0

def creatmodel():
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Flatten(input_shape=(28, 28)));
	model.add(tf.keras.layers.Dense(320, activation='relu'));
	model.add(tf.keras.layers.Dense(10));
	model.compile(keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['sparse_categorical_accuracy'])
	return model

#model = creatmodel()
#model.fit(train_images, train_labels, epochs=26,validation_data=(test_images,test_labels))
#model.save('./cloth_model2.h5')

#new_model = tf.keras.models.load_model('./cloth_model2.h5')
#res = new_model.evaluate(test_images, test_labels, verbose=2)
#print(res)

print(test_labels)

print(test_images.shape)

#imgarray = []
#imgarray.append(test_images[0].tolist());
#imgarray.append(test_images[1].tolist());
#img2 = np.array(imgarray)
#print(img2.shape)
#print(img2)

img1 = test_images[0].copy()
print(img1.shape)
img1 = np.vstack((img1[None],test_images[2][None]))
print(img1.shape)
img1 = np.vstack((img1,test_images[3][None]))
print(img1.shape)
img1 = np.vstack((img1,test_images[4][None]))
print(img1.shape)

