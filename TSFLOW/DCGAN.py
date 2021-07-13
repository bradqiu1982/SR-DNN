import tensorflow as tf
import traceback
import contextlib

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import random

import os
import seaborn as sns

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import models

from PIL import Image
import time

#from IPython import display


# # Fetch and format the mnist data
# (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()
# dataset = tf.data.Dataset.from_tensor_slices((tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),tf.cast(mnist_labels,tf.int64)))
# dataset = dataset.shuffle(1000).batch(32)

# # Build the model
# mnist_model = tf.keras.Sequential([
# 	tf.keras.layers.Conv2D(16,[3,3], activation='relu',input_shape=(None, None, 1)),
# 	tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
# 	tf.keras.layers.GlobalAveragePooling2D(),
# 	tf.keras.layers.Dense(10)
# ])

# mnist_model.summary()

# optimizer = tf.keras.optimizers.Adam()
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# #loss_history = []


# def train_step(images, labels):
# 	with tf.GradientTape() as tape:
# 		logits = mnist_model(images, training=True)
# 		# Add asserts to check the shape of the output.
# 		#tf.debugging.assert_equal(logits.shape, (32, 10))
# 		loss_value = loss_object(labels, logits)

# 	#loss_history.append(loss_value.numpy().mean())
# 	grads = tape.gradient(loss_value, mnist_model.trainable_variables)
# 	optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))

# def train(epochs):
# 	for epoch in range(epochs):
# 		for (batch, (images, labels)) in enumerate(dataset):
# 			train_step(images, labels)
# 		print ('Epoch {} finished'.format(epoch))
# train(epochs = 1)






# class Dense(tf.Module):
#   def __init__(self, in_features, out_features, name=None):
#     super().__init__(name=name)
#     self.w = tf.Variable(
#       tf.random.normal([in_features, out_features]), name='w')
#     self.b = tf.Variable(tf.zeros([out_features]), name='b')
#   def __call__(self, x):
#     y = tf.matmul(x, self.w) + self.b
#     return tf.nn.relu(y)

# class SequentialModule(tf.Module):
#   def __init__(self, name=None):
#     super().__init__(name=name)

#     self.dense_1 = Dense(in_features=3, out_features=3)
#     self.dense_2 = Dense(in_features=3, out_features=2)

#   def __call__(self, x):
#     x = self.dense_1(x)
#     return self.dense_2(x)

# # You have made a model!
# my_model = SequentialModule(name="the_model")

# # Call it, with random results
# print("Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))



def APIModel():
	inputs = keras.Input(shape=(28,28,1))
	x = layers.Dense(64, activation="relu")(inputs)

	x = layers.Conv2D(64, (3, 3), activation='relu')(x)
	x = layers.MaxPooling2D((2, 2))(x)
	x = layers.Dense(64, activation="relu")(x)
	x = layers.Conv2D(64, (3, 3), activation='relu')(x)
	x = layers.MaxPooling2D((2, 2))(x)
	x = layers.Dense(64, activation="relu")(x)
	x = layers.Flatten()(x)
	outputs = layers.Dense(10, activation='softmax')(x)
	model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
	model.summary()

	(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

	print(x_train.shape)
	print(y_train.shape)

	x_train = x_train.reshape(60000, 28,28,1).astype("float32") / 255
	x_test = x_test.reshape(10000, 28,28,1).astype("float32") / 255


	model.compile(
	    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	    optimizer=keras.optimizers.RMSprop(),
	    metrics=["accuracy"],
	)

	print(x_train.shape)

	history = model.fit(x_train, y_train, epochs=5, validation_split=0.2)

	test_scores = model.evaluate(x_test, y_test, verbose=2)
	print("Test loss:", test_scores[0])
	print("Test accuracy:", test_scores[1])
	model.save('./cnn_model.h5')


def ResNetModel():
	inputs = keras.Input(shape=(28, 28, 1), name="img")
	x = layers.Conv2D(32, 3, activation="relu")(inputs)
	x = layers.Conv2D(64, 3, activation="relu")(x)
	block_1_output = layers.MaxPooling2D(3)(x)

	x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
	x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
	block_2_output = layers.add([x, block_1_output])

	x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
	x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
	block_25_output = layers.add([x, block_2_output])

	x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_25_output)
	x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
	block_3_output = layers.add([x, block_25_output])

	x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
	x = layers.GlobalAveragePooling2D()(x)
	x = layers.Dense(256, activation="relu")(x)
	x = layers.Dropout(0.2)(x)
	outputs = layers.Dense(10, activation='softmax')(x)

	model = keras.Model(inputs, outputs, name="toy_resnet")
	model.summary()

	(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


	x_train = x_train.reshape(60000, 28,28,1).astype("float32") / 255
	x_test = x_test.reshape(10000, 28,28,1).astype("float32") / 255


	model.compile(
	    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	    optimizer=keras.optimizers.RMSprop(),
	    metrics=["sparse_categorical_accuracy"],
	)

	#history = model.fit(x_train, y_train, epochs=3, validation_split=0.2)

	train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

	callbacks = [keras.callbacks.ModelCheckpoint(
		filepath='./train_ckpt/ckpt-{epoch}',
		verbose=1, 
    	save_weights_only=True
	)]


	# data_root = pathlib.Path('./train_ckpt')
	# all_ckpt_paths = list(data_root.glob('*'))
	# all_ckpt_paths = [str(path) for path in all_ckpt_paths]
	# ckptexist = False
	# for p in all_ckpt_paths:
	# 	if '.index' in p:
	# 		ckptexist = True

	# if ckptexist:
	# 	latest = tf.train.latest_checkpoint('./train_ckpt')
	# 	model.load_weights(latest)


	model.fit(train_dataset, epochs=3,callbacks=callbacks)

	test_scores = model.evaluate(x_test, y_test, verbose=2)
	print("Test loss:", test_scores[0])
	print("Test accuracy:", test_scores[1])
	model.save('./rest_model.h5')

def RNNModel():
	batch_size = 64
	input_dim = 28
	units = 64
	output_size  = 10

	model = keras.Sequential()
	model.add(layers.LSTM(units, input_shape=(None, input_dim),  return_sequences=True))
	model.add(layers.Bidirectional(layers.LSTM(32)))
	model.add(layers.BatchNormalization())
	model.add(layers.Dense(output_size))
	model.summary()

	mnist = keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0
	sample, sample_label = x_train[0], y_train[0]


	model.compile(
	    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	    optimizer="sgd",
	    metrics=["sparse_categorical_accuracy"],
	)

	model.fit(
	    x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=10
	)

	#result = tf.argmax(model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)

	result = tf.argmax(model.predict(tf.expand_dims(sample, 0)), axis=1)

	print( "Predicted result is:")
	print(result.numpy())
	print("target result is:")
	print(sample_label)




AUTOTUNE = tf.data.AUTOTUNE

def decode_audio(audio_binary):
	audio, _ = tf.audio.decode_wav(audio_binary)
	return tf.squeeze(audio, axis=-1)

def get_waveform(file_path):
	audio_binary = tf.io.read_file(file_path)
	waveform = decode_audio(audio_binary)
	return waveform

def get_spectrogram_(waveform):
	# Padding for files with less than 16000 samples
	zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
	# Concatenate audio with padding so that all audio clips will be of the 
	# same length
	waveform = tf.cast(waveform, tf.float32)
	equal_length = tf.concat([waveform, zero_padding], 0)
	spectrogram = tf.signal.stft( equal_length, frame_length=255, frame_step=128)
	spectrogram = tf.abs(spectrogram)
	return spectrogram


def get_spectrogram_IMG(audio):
	spectrogram = get_spectrogram_(audio)
	spectrogram = tf.expand_dims(spectrogram, -1)
	spectrogram = tf.image.resize(spectrogram ,[128,128])
	return spectrogram

def preprocess_dataset_IMG(files,labelsds):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(get_spectrogram_IMG,  num_parallel_calls=AUTOTUNE)
  return tf.data.Dataset.zip((output_ds,labelsds))



def get_spectrogram_RNN(audio):
	spectrogram = get_spectrogram_(audio)
	spectrogram = tf.expand_dims(spectrogram, -1)
	spectrogram = tf.image.resize(spectrogram ,[64,64])
	return tf.squeeze(spectrogram)

def preprocess_dataset_RNN(files,labelsds):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(get_spectrogram_RNN,  num_parallel_calls=AUTOTUNE)
  return tf.data.Dataset.zip((output_ds,labelsds))

def voiceTrain_RNN():
	data_root = pathlib.Path('./AIdata/mini_speech_commands')
	all_image_paths = list(data_root.glob('*/*'))
	all_image_paths = [str(path) for path in all_image_paths]
	random.shuffle(all_image_paths)


	label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
	label_to_index = dict((name, index) for index, name in enumerate(label_names))

	train_files = all_image_paths[:6400]
	val_files = all_image_paths[6400: 6400 + 800]
	test_files = all_image_paths[-800:]

	train_labels = [label_to_index[pathlib.Path(path).parent.name] for path in train_files]
	val_labels = [label_to_index[pathlib.Path(path).parent.name] for path in val_files]
	test_labels = [label_to_index[pathlib.Path(path).parent.name] for path in test_files]

	train_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.int64))
	val_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val_labels, tf.int64))
	test_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(test_labels, tf.int64))


	spectrogram_ds = preprocess_dataset_RNN(train_files,train_labels_ds)
	train_ds = spectrogram_ds
	val_ds  = preprocess_dataset_RNN(val_files,val_labels_ds)
	test_ds = preprocess_dataset_RNN(test_files,test_labels_ds)

	batch_size = 64
	train_ds = train_ds.batch(batch_size)
	val_ds = val_ds.batch(batch_size)

	# train_ds = train_ds.cache().prefetch(AUTOTUNE)
	# val_ds = val_ds.cache().prefetch(AUTOTUNE)


	for spectrogram, _ in spectrogram_ds.take(1):
		input_shape = spectrogram.shape
	print('Input shape:', input_shape)


	num_labels = len(label_names)
	norm_layer = preprocessing.Normalization()
	norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

	model = keras.Sequential()
	model.add(layers.Input(shape=input_shape))
	model.add(norm_layer)
	model.add(layers.LSTM(64, input_shape=(None, 64), return_sequences=True))
	model.add(layers.Bidirectional(layers.LSTM(32)))
	model.add(layers.BatchNormalization())
	# model.add(layers.Dense(32, activation="relu"))
	# model.add(layers.Flatten())
	model.add(layers.Dense(num_labels))
	model.summary()


	model.compile(
	    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	    optimizer="sgd",
	    metrics=["accuracy"],
	)

	history = model.fit(train_ds,validation_data=val_ds,epochs=10)


def voiceTrain_CNN():
	data_root = pathlib.Path('./AIdata/mini_speech_commands')
	all_image_paths = list(data_root.glob('*/*'))
	all_image_paths = [str(path) for path in all_image_paths]
	random.shuffle(all_image_paths)


	label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
	label_to_index = dict((name, index) for index, name in enumerate(label_names))

	train_files = all_image_paths[:6400]
	val_files = all_image_paths[6400: 6400 + 800]
	test_files = all_image_paths[-800:]

	train_labels = [label_to_index[pathlib.Path(path).parent.name] for path in train_files]
	val_labels = [label_to_index[pathlib.Path(path).parent.name] for path in val_files]
	test_labels = [label_to_index[pathlib.Path(path).parent.name] for path in test_files]

	train_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.int64))
	val_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val_labels, tf.int64))
	test_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(test_labels, tf.int64))


	spectrogram_ds = preprocess_dataset_IMG(train_files,train_labels_ds)
	train_ds = spectrogram_ds
	val_ds  = preprocess_dataset_IMG(val_files,val_labels_ds)
	test_ds = preprocess_dataset_IMG(test_files,test_labels_ds)

	batch_size = 64
	train_ds = train_ds.batch(batch_size)
	val_ds = val_ds.batch(batch_size)

	# train_ds = train_ds.cache().prefetch(AUTOTUNE)
	# val_ds = val_ds.cache().prefetch(AUTOTUNE)


	for spectrogram, _ in spectrogram_ds.take(1):
		input_shape = spectrogram.shape
	print('Input shape:', input_shape)


	num_labels = len(label_names)
	norm_layer = preprocessing.Normalization()
	norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

	model = models.Sequential([
		layers.Input(shape=input_shape),
		norm_layer,
		layers.Conv2D(32, 3, activation='relu'),
		layers.Conv2D(64, 3, activation='relu'),
		layers.MaxPooling2D(),
		layers.Flatten(),
		layers.Dense(128, activation='relu'),
		layers.Dropout(0.2),
		layers.Dense(num_labels, activation='softmax'),
		])

	model.summary()
	model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
	#history = model.fit(train_ds,validation_data=val_ds,epochs=10,callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2))
	history = model.fit(train_ds,validation_data=val_ds,epochs=10)


def voiceTrain_RESNET():
	data_root = pathlib.Path('./AIdata/mini_speech_commands')
	all_image_paths = list(data_root.glob('*/*'))
	all_image_paths = [str(path) for path in all_image_paths]
	random.shuffle(all_image_paths)


	label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
	label_to_index = dict((name, index) for index, name in enumerate(label_names))

	train_files = all_image_paths[:6400]
	val_files = all_image_paths[6400: 6400 + 800]
	test_files = all_image_paths[-800:]

	train_labels = [label_to_index[pathlib.Path(path).parent.name] for path in train_files]
	val_labels = [label_to_index[pathlib.Path(path).parent.name] for path in val_files]
	test_labels = [label_to_index[pathlib.Path(path).parent.name] for path in test_files]

	train_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels, tf.int64))
	val_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val_labels, tf.int64))
	test_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(test_labels, tf.int64))


	spectrogram_ds = preprocess_dataset_IMG(train_files,train_labels_ds)
	train_ds = spectrogram_ds
	val_ds  = preprocess_dataset_IMG(val_files,val_labels_ds)
	test_ds = preprocess_dataset_IMG(test_files,test_labels_ds)

	batch_size = 64
	train_ds = train_ds.batch(batch_size)
	val_ds = val_ds.batch(batch_size)

	train_ds = train_ds.cache().prefetch(AUTOTUNE)
	val_ds = val_ds.cache().prefetch(AUTOTUNE)


	for spectrogram, _ in spectrogram_ds.take(1):
		input_shape = spectrogram.shape
	print('Input shape:', input_shape)


	num_labels = len(label_names)
	norm_layer = preprocessing.Normalization()
	norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

	inputs = keras.Input(shape=input_shape, name="img")
	x = norm_layer(inputs)

	x = layers.Conv2D(32, 3, activation="relu")(x)
	x = layers.Conv2D(64, 3, activation="relu")(x)
	block_1_output = layers.MaxPooling2D(3)(x)

	x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
	x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
	block_2_output = layers.add([x, block_1_output])

	x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
	x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
	block_25_output = layers.add([x, block_2_output])

	x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_25_output)
	x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
	block_3_output = layers.add([x, block_25_output])

	x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
	x = layers.GlobalAveragePooling2D()(x)
	x = layers.Dense(256, activation="relu")(x)
	x = layers.Dropout(0.2)(x)
	outputs = layers.Dense(num_labels, activation='softmax')(x)

	model = keras.Model(inputs, outputs, name="toy_resnet")
	model.summary()

	model.compile(
	    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	    optimizer=keras.optimizers.RMSprop(),
	    metrics=["sparse_categorical_accuracy"],
	)

	history = model.fit(train_ds,validation_data=val_ds,epochs=10)


def make_generator_model():
	model = tf.keras.Sequential()

	model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Reshape((7, 7, 256)))

	model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))

	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))

	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())

	model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

	return model


def make_discriminator_model():
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[28, 28, 1]))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Flatten())
	model.add(layers.Dense(1))

	return model


def WriteChar(generated_image,ix):
	generated_image = generated_image*127.5+127.5;
	ia = tf.squeeze(generated_image).numpy()
	im = Image.fromarray(ia)
	if im.mode == 'F':
		im = im.convert('RGB')
	im.resize((60,60))
	im.save('./gen/noise'+str(ix)+'.jpg')


def DCGAN():
	(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
	train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
	train_images = (train_images - 127.5)/127.5
	
	BUFFER_SIZE = 60000
	BATCH_SIZE = 256
	train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

	generator = make_generator_model()

	discriminator = make_discriminator_model()

	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

	def discriminator_loss(real_output, fake_output):
		real_loss = cross_entropy(tf.ones_like(real_output), real_output)
		fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
		total_loss = real_loss + fake_loss
		return total_loss

	def generator_loss(fake_output):
		return cross_entropy(tf.ones_like(fake_output), fake_output)

	generator_optimizer = tf.keras.optimizers.Adam(1e-4)
	discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


	checkpoint_dir = './training_checkpoints'
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,generator=generator,discriminator=discriminator)
	ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
	if ckpt_manager.latest_checkpoint:
		checkpoint.restore(ckpt_manager.latest_checkpoint)
		print ('Latest checkpoint restored!!')


	EPOCHS = 1
	noise_dim = 100
	num_examples_to_generate = 16

	seed = tf.random.normal([num_examples_to_generate, noise_dim])


	@tf.function
	def train_step(images):
		noise = tf.random.normal([BATCH_SIZE, noise_dim])

		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			generated_images = generator(noise, training=True)

			real_output = discriminator(images, training=True)
			fake_output = discriminator(generated_images, training=True)

			gen_loss = generator_loss(fake_output)
			disc_loss = discriminator_loss(real_output, fake_output)

		gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
		gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

		generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
		discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


	def train(dataset, epochs):
		for epoch in range(epochs):
			start = time.time()

			for image_batch in dataset:
				train_step(image_batch)

			print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

			if (epoch + 1) % 3 == 0:
				ckpt_save_path = ckpt_manager.save()
				print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,ckpt_save_path))

		predictions = generator(seed, training=False)
		for i in range(predictions.shape[0]):
			WriteChar(predictions[i, :, :, 0],i)

	train(train_dataset, EPOCHS)
	generator.save('./handwrite_generator.h5')
	discriminator.save('./handwrite_discriminator.h5')



def GenerateVerify():
	generator = tf.keras.models.load_model('./handwrite_generator.h5')
	noise_dim = 100
	num_examples_to_generate = 16
	seed = tf.random.normal([num_examples_to_generate, noise_dim])
	predictions = generator(seed, training=False)
	for i in range(predictions.shape[0]):
		WriteChar(predictions[i, :, :, 0],i)



DCGAN()

#GenerateVerify()

#APIModel()

#ResNetModel()

#RNNModel()


#voiceTrain_RESNET()


