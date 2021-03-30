
import tensorflow as tf
import pathlib
import random
import matplotlib.pyplot as plt
import os
from PIL import Image

import tensorflow_hub as hub

import numpy as np
from tensorflow import keras

SZ = 224
#FILEFOLDER = '\\\\wux-engsys01\\PlanningForCast\\VMI'
FILEFOLDER = '\\\\wux-engsys01\\PlanningForCast\\flowers'
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32



def preprocess_image(image_raw):
	img_tensor  = tf.image.decode_jpeg(image_raw, channels=3)
	img_tensor  = tf.image.resize(img_tensor , [SZ, SZ])
	img_tensor  /= 255.0  # normalize to [0,1] range
	return img_tensor

def load_and_preprocess_image(path):
	image = tf.io.read_file(path)
	return preprocess_image(image)


def change_range(image,label):
		return 2*image-1, label

def get_training_ds():
	data_root = pathlib.Path(FILEFOLDER)

	all_image_paths = list(data_root.glob('*/*'))
	all_image_paths = [str(path) for path in all_image_paths]
	random.shuffle(all_image_paths)
	label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
	label_to_index = dict((name, index) for index, name in enumerate(label_names))
	all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]


	path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
	image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
	label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
	image_count = len(all_image_paths)
	image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))



	ds = image_label_ds.shuffle(buffer_size=image_count)
	ds = ds.repeat()
	ds = ds.batch(BATCH_SIZE)
	ds = ds.prefetch(buffer_size=AUTOTUNE)
	
	return (ds,image_count,len(label_names))


def get_training_dsonehot():
	data_root = pathlib.Path(FILEFOLDER)

	all_image_paths = list(data_root.glob('*/*'))
	all_image_paths = [str(path) for path in all_image_paths]
	random.shuffle(all_image_paths)
	label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())


	labels  = [i for i in range(0,len(label_names))]
	one_hot_index = np.arange(len(labels)) * len(labels) + labels
	one_hot = np.zeros((len(labels), len(labels)))
	one_hot.flat[one_hot_index] = 1
	label_to_index ={}
	for i in range(0,len(label_names)):
		label_to_index[label_names[i]] = one_hot[i]


	all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]


	path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
	image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
	label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
	image_count = len(all_image_paths)
	image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))


	ds = image_label_ds.shuffle(buffer_size=image_count)
	ds = ds.repeat()
	ds = ds.batch(BATCH_SIZE)
	ds = ds.prefetch(buffer_size=AUTOTUNE)
	
	return (ds,image_count,len(label_names))

def train_by_self():
	ds,image_count,classcnt = get_training_ds()
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(SZ, SZ, 3)))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(64, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2));
	model.add(tf.keras.layers.Dense(classcnt, activation='softmax'))

	model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['sparse_categorical_accuracy'])

	steps_per_epoch=tf.math.ceil((image_count+2)/BATCH_SIZE).numpy()
	model.fit(ds, epochs=2,steps_per_epoch=steps_per_epoch)
	model.save('./FLOWER_CLASS_self.h5')


def train_by_mobilev3_hub():
	os.environ["TFHUB_CACHE_DIR"] = "./hub_model"
	ds,image_count,classcnt = get_training_ds()
	keras_ds = ds.map(change_range)
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.InputLayer(input_shape=(454,454,3)))
	model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(3, (3, 3), activation='relu'))
	model.add(hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5", trainable=True))
	model.add(tf.keras.layers.Dropout(rate=0.2))
	model.add(tf.keras.layers.Dense(classcnt, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
	model.trainable=True

	model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["sparse_categorical_accuracy"])
	model.summary()
	#steps_per_epoch=tf.math.ceil(image_count/BATCH_SIZE).numpy()
	#model.fit(keras_ds,epochs=3,steps_per_epoch=steps_per_epoch)
	#model.save('./VCSEL_CLASS_mobilev3.h5')

def train_by_vgg19():
	ds,image_count,classcnt = get_training_ds()

	model = tf.keras.models.Sequential();
	model.add(tf.keras.applications.VGG19(include_top=False,input_shape=(SZ,SZ,3)))#, classes=classcnt))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(1024, activation='relu',input_dim=512))
	model.add(tf.keras.layers.Dense(512, activation='relu'))
	model.add(tf.keras.layers.Dense(256, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2));
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2));
	model.add(tf.keras.layers.Dense(classcnt, activation='softmax'))
	model.trainable=True

	#model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
	model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.9), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['sparse_categorical_accuracy'])
	
	model.summary()
	steps_per_epoch=tf.math.ceil(image_count/BATCH_SIZE).numpy()
	model.fit(ds,epochs=3,steps_per_epoch=steps_per_epoch)
	model.save('./FLOWER_CLASS_VGG19.h5')
	

def VerifySelfModel():
	data_root = pathlib.Path(FILEFOLDER)
	all_image_paths = list(data_root.glob('*/*'))
	all_image_paths = [str(path) for path in all_image_paths]
	random.shuffle(all_image_paths)
	label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
	label_to_index = dict((index, name) for index, name in enumerate(label_names))

	img = tf.io.read_file('\\\\wux-engsys01\\PlanningForCast\\VMI\\broken\\芯片断裂2.jpg')
	image_tensor = tf.io.decode_image(img, channels=3)
	image_tensor  = tf.image.resize(image_tensor , [SZ, SZ])
	#image_tensor = tf.cast(image_tensor,tf.float32)
	image_tensor  /= 255.0
	#image_tensor = tf.cast(image_tensor,tf.uint8);
	image_tensor = tf.expand_dims(image_tensor, axis=0)

	model = tf.keras.models.load_model('./VCSEL_CLASS_self.h5')
	res = model.predict(image_tensor)
	print(res.flatten())
	mxidx = np.argmax(res.flatten())
	print(label_to_index[mxidx])

def VerifyVGG19Model():
	data_root = pathlib.Path(FILEFOLDER)
	all_image_paths = list(data_root.glob('*/*'))
	all_image_paths = [str(path) for path in all_image_paths]
	random.shuffle(all_image_paths)
	label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
	label_to_index = dict((index, name) for index, name in enumerate(label_names))

	img = tf.io.read_file('\\\\wux-engsys01\\PlanningForCast\\flowers\\sunflowers\\184683023_737fec5b18.jpg')
	image_tensor = tf.io.decode_image(img, channels=3)
	image_tensor  = tf.image.resize(image_tensor , [SZ, SZ])
	#image_tensor = tf.cast(image_tensor,tf.float32)
	image_tensor  /= 255.0
	#image_tensor = tf.cast(image_tensor,tf.uint8);
	image_tensor = tf.expand_dims(image_tensor, axis=0)

	model = tf.keras.models.load_model('./FLOWER_CLASS_VGG19.h5')
	res = model.predict(image_tensor)
	print(res.flatten())
	mxidx = np.argmax(res.flatten())
	print(label_to_index[mxidx])


#train_by_self()
#VerifySelfModel()
#train_by_vgg19()
train_by_mobilev3_hub()
