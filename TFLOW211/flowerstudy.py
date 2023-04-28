import tensorflow as tf
import pathlib
import random
import matplotlib.pyplot as plt
import os
from PIL import Image

import tensorflow_hub as hub

import numpy as np

def preprocess_image(image_raw):
	img_tensor  = tf.image.decode_jpeg(image_raw, channels=3)
	img_tensor  = tf.image.resize(img_tensor , [192, 192])
	img_tensor  /= 255.0  # normalize to [0,1] range
	return img_tensor

def load_and_preprocess_image(path):
	image = tf.io.read_file(path)
	return preprocess_image(image)

def change_range(image,label):
		return 2*image-1, label

def preprocess_image224(image_raw):
	img_tensor  = tf.image.decode_jpeg(image_raw, channels=3)
	img_tensor  = tf.image.resize(img_tensor , [224, 224])
	img_tensor  /= 255.0  # normalize to [0,1] range
	return img_tensor

def load_and_preprocess_image224(path):
	image = tf.io.read_file(path)
	return preprocess_image224(image)

def training1():
	AUTOTUNE = tf.data.experimental.AUTOTUNE
	data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',fname='flower_photos', untar=True)
	data_root = pathlib.Path(data_root_orig)


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


	#ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
	#def load_and_preprocess_from_path_label(path, label):
	#	return load_and_preprocess_image(path), label
	#image_label_ds = ds.map(load_and_preprocess_from_path_label)

	
	BATCH_SIZE = 32
	ds = image_label_ds.shuffle(buffer_size=image_count)
	ds = ds.repeat()
	ds = ds.batch(BATCH_SIZE)
	ds = ds.prefetch(buffer_size=AUTOTUNE)
	keras_ds = ds.map(change_range)


	#ds = image_label_ds.cache()
	#ds = ds.apply( tf.data.experimental.shuffle(buffer_size=image_count))
	#ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

	mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
	mobile_net.trainable=False

	model = tf.keras.Sequential([ mobile_net, tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(len(label_names), activation = 'softmax')])
	model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=["sparse_categorical_accuracy"])


	checkpoint_path = "training_5\\cp.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,verbose=1)

	steps_per_epoch=tf.math.ceil(image_count/BATCH_SIZE).numpy()
	model.fit(keras_ds,epochs=5,steps_per_epoch=steps_per_epoch,callbacks=[cp_callback])

	model.save('./flower_model_v2_wt5.h5')

	#new_model = tf.keras.models.load_model('./flower_model.h5')
	#res = new_model.evaluate(test_images, test_labels, verbose=2)
	#print(res)


def training2():
	AUTOTUNE = tf.data.experimental.AUTOTUNE
	data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',fname='flower_photos', untar=True)
	data_root = pathlib.Path(data_root_orig)


	all_image_paths = list(data_root.glob('*/*'))
	all_image_paths = [str(path) for path in all_image_paths]
	random.shuffle(all_image_paths)
	label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

	labels  = [i for i in range(0,len(label_names))]
	one_hot_index = np.arange(len(labels)) * len(labels) + labels
	one_hot = np.zeros((len(labels), len(labels)))
	one_hot.flat[one_hot_index] = 1
	label_to_index ={}
	for i in range(0,5):
		label_to_index[label_names[i]] = one_hot[i]

	all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]


	path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
	image_ds = path_ds.map(load_and_preprocess_image224, num_parallel_calls=AUTOTUNE)
	label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
	image_count = len(all_image_paths)
	image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

	print(image_ds)

	#ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
	#def load_and_preprocess_from_path_label(path, label):
	#	return load_and_preprocess_image(path), label
	#image_label_ds = ds.map(load_and_preprocess_from_path_label)

	
	BATCH_SIZE = 32
	ds = image_label_ds.shuffle(buffer_size=image_count)
	ds = ds.repeat()
	ds = ds.batch(BATCH_SIZE)
	ds = ds.prefetch(buffer_size=AUTOTUNE)
	keras_ds = ds.map(change_range)


	#mobile_net = tf.keras.applications.MobileNetV3Small()

	model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(224,224,3)), hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5", trainable=True), tf.keras.layers.Dropout(rate=0.2), tf.keras.layers.Dense(len(label_names), kernel_regularizer=tf.keras.regularizers.l2(0.0001))])
	model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1), metrics=["accuracy"])
	model.summary()

	steps_per_epoch=tf.math.ceil(image_count/BATCH_SIZE).numpy()
	model.fit(keras_ds,epochs=3,steps_per_epoch=steps_per_epoch)
	model.save('./flower_model_v3_wt3.h5')

def training3():
	AUTOTUNE = tf.data.experimental.AUTOTUNE
	data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',fname='flower_photos', untar=True)
	data_root = pathlib.Path(data_root_orig)


	all_image_paths = list(data_root.glob('*/*'))
	all_image_paths = [str(path) for path in all_image_paths]
	random.shuffle(all_image_paths)
	label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
	label_to_index = dict((name, index) for index, name in enumerate(label_names))
	all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]


	path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
	image_ds = path_ds.map(load_and_preprocess_image224, num_parallel_calls=AUTOTUNE)
	label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
	image_count = len(all_image_paths)
	image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

	#print(image_ds)

	#ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
	#def load_and_preprocess_from_path_label(path, label):
	#	return load_and_preprocess_image(path), label
	#image_label_ds = ds.map(load_and_preprocess_from_path_label)

	
	BATCH_SIZE = 32
	ds = image_label_ds.shuffle(buffer_size=image_count)
	ds = ds.repeat()
	ds = ds.batch(BATCH_SIZE)
	ds = ds.prefetch(buffer_size=AUTOTUNE)
	keras_ds = ds.map(change_range)


	#mobile_net = tf.keras.applications.MobileNetV3Small()

	model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(224,224,3)), hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5", trainable=True), tf.keras.layers.Dropout(rate=0.2), tf.keras.layers.Dense(len(label_names), kernel_regularizer=tf.keras.regularizers.l2(0.0001))])
	model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["sparse_categorical_accuracy"])
	model.summary()

	steps_per_epoch=tf.math.ceil(image_count/BATCH_SIZE).numpy()
	model.fit(keras_ds,epochs=3,steps_per_epoch=steps_per_epoch)
	model.save('./flower_model_v3_wt4.h5')

def verifying4training2():
	AUTOTUNE = tf.data.experimental.AUTOTUNE
	data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',fname='flower_photos', untar=True)
	data_root = pathlib.Path(data_root_orig)


	all_image_paths = list(data_root.glob('*/*'))
	all_image_paths = [str(path) for path in all_image_paths]
	random.shuffle(all_image_paths)
	all_image_paths = all_image_paths[:20]

	label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

	#label_to_index = dict((name, index) for index, name in enumerate(label_names))
	#print(label_to_index)

	labels  = [i for i in range(0,len(label_names))]
	one_hot_index = np.arange(len(labels)) * len(labels) + labels
	one_hot = np.zeros((len(labels), len(labels)))
	one_hot.flat[one_hot_index] = 1
	label_to_index ={}
	for i in range(0,5):
		label_to_index[label_names[i]] = one_hot[i]

	all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
	all_image = [load_and_preprocess_image224(path) for path in all_image_paths]
	all_images = [2*img-1 for img in all_image]


	image_ds = np.array(all_images);
	label_ds = np.array(all_image_labels);

	#print(label_ds)
	#print(image_ds.shape)
	# path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
	# image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
	# label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

	# image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
	# keras_ds = image_label_ds.map(change_range)

	#new_model = tf.keras.models.load_model('./flower_model_v3_wt3.h5')
	new_model = tf.keras.models.load_model('./flower_model_v3_wt3.h5', custom_objects={'KerasLayer': hub.KerasLayer})
	res = new_model.evaluate(image_ds,label_ds, verbose=2)
	print(res)

	i = image_ds[8:9].reshape(224,224,3)
	i = (i+1.0)/2.0
	i = i*255.0
	i = i.astype(np.uint8)
	im = Image.fromarray(i)
	im.save('./myflower.png')

	res = new_model.predict(image_ds[8:9])
	print(res.flatten())

def verifying4training1():
	AUTOTUNE = tf.data.experimental.AUTOTUNE
	data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',fname='flower_photos', untar=True)
	data_root = pathlib.Path(data_root_orig)


	all_image_paths = list(data_root.glob('*/*'))
	all_image_paths = [str(path) for path in all_image_paths]
	random.shuffle(all_image_paths)
	all_image_paths = all_image_paths[:20]


	label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
	label_to_index = dict((name, index) for index, name in enumerate(label_names))
	all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]


	path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
	image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
	label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))


	image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
	image_label_ds = image_label_ds.batch(32)
	image_label_ds = image_label_ds.map(change_range)

	new_model = tf.keras.models.load_model('./flower_model_v2_def.h5')
	res = new_model.evaluate(image_label_ds, verbose=2)
	print(res)


def verifying4training3():
	AUTOTUNE = tf.data.experimental.AUTOTUNE
	data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',fname='flower_photos', untar=True)
	data_root = pathlib.Path(data_root_orig)


	all_image_paths = list(data_root.glob('*/*'))
	all_image_paths = [str(path) for path in all_image_paths]
	random.shuffle(all_image_paths)
	all_image_paths = all_image_paths[:20]

	label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
	label_to_index = dict((name, index) for index, name in enumerate(label_names))

	all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
	all_image = [load_and_preprocess_image224(path) for path in all_image_paths]
	all_images = [2*img-1 for img in all_image]


	image_ds = np.array(all_images);
	label_ds = np.array(all_image_labels);

	#print(label_ds)
	#print(image_ds.shape)

	#new_model = tf.keras.models.load_model('./flower_model_v3_wt3.h5')
	new_model = tf.keras.models.load_model('./flower_model_v3_wt4.h5', custom_objects={'KerasLayer': hub.KerasLayer})
	res = new_model.evaluate(image_ds,label_ds, verbose=2)
	print(res)

	i = image_ds[8:9].reshape(224,224,3)
	i = (i+1.0)/2.0
	i = i*255.0
	i = i.astype(np.uint8)
	im = Image.fromarray(i)
	im.save('./myflower.png')

	res = new_model.predict(image_ds[8:9])
	print(res.flatten())


verifying4training3()
