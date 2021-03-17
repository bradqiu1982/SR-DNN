
from PIL import Image
import base64
import io
import numpy as np
import pyodbc

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import cv2

def convtimg(s64):
	bt = base64.b64decode(s64)
	im = Image.open(io.BytesIO(bt))
	img1 = np.array(im)
	img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
	img1 = img1/255.0
	return img1

def getTrainData():
	AUTOTUNE = tf.data.experimental.AUTOTUNE
	imgstr = []
	labs = []

	with pyodbc.connect(Driver='{ODBC Driver 17 for SQL Server}',Server='wuxinpi.china.ads.finisar.com', UID='WATApp', PWD='WATApp@123', Database='WAT') as conn:
		cursor = conn.cursor()
		sql = "select top 2 [TrainingImg],[ImgVal] from [WAT].[dbo].[AITrainingData] where Revision = 'OGP-rect5x1'"
		cursor.execute(sql)
		rows = cursor.fetchall() 
		for row in rows:
			imgstr.append(str(row[0]))
			labs.append(int(row[1])-48)
		cursor.close()

	img1 = convtimg(imgstr[0])
	img2 = convtimg(imgstr[1])
	imgarray = np.vstack((img1[None],img2[None]))

	#print(img2.shape)
	#print(imgarray.shape)

	print('start loading data..........')

	with pyodbc.connect(Driver='{ODBC Driver 17 for SQL Server}',Server='wuxinpi.china.ads.finisar.com', UID='WATApp', PWD='WATApp@123', Database='WAT') as conn:
		cursor = conn.cursor()
		sql = "select top 5000 [TrainingImg],[ImgVal] from [WAT].[dbo].[AITrainingData] where Revision = 'OGP-rect5x1' order by UpdateTime desc"
		cursor.execute(sql)
		rows = cursor.fetchall() 
		for row in rows:
			img2 = convtimg(str(row[0]))
			imgarray = np.vstack((imgarray,img2[None]))
			labs.append(int(row[1])-48)
		cursor.close()

	labarray = np.array(labs)

	label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labarray, tf.int64))
	image_ds = tf.data.Dataset.from_tensor_slices(imgarray)
	train_dataset = tf.data.Dataset.zip((image_ds, label_ds))

	train_dataset = train_dataset.repeat()
	BATCH_SIZE = 32
	SHUFFLE_BUFFER_SIZE = 500
	train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
	train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

	return train_dataset


def getTestData():
	imgstr = []
	labs = []

	with pyodbc.connect(Driver='{ODBC Driver 17 for SQL Server}',Server='wuxinpi.china.ads.finisar.com', UID='WATApp', PWD='WATApp@123', Database='WAT') as conn:
		cursor = conn.cursor()
		sql = "select top 2 [TrainingImg],[ImgVal] from [WAT].[dbo].[AITrainingData] where Revision = 'OGP-rect5x1'"
		cursor.execute(sql)
		rows = cursor.fetchall() 
		for row in rows:
			imgstr.append(str(row[0]))
			labs.append(int(row[1])-48)
		cursor.close()

	img1 = convtimg(imgstr[0])
	img2 = convtimg(imgstr[1])
	imgarray = np.vstack((img1[None],img2[None]))


	with pyodbc.connect(Driver='{ODBC Driver 17 for SQL Server}',Server='wuxinpi.china.ads.finisar.com', UID='WATApp', PWD='WATApp@123', Database='WAT') as conn:
		cursor = conn.cursor()
		sql = "select top 40 [TrainingImg],[ImgVal] from [WAT].[dbo].[AITrainingData] where Revision = 'OGP-rect5x1' order by UpdateTime asc"
		cursor.execute(sql)
		rows = cursor.fetchall() 
		for row in rows:
			img2 = convtimg(str(row[0]))
			imgarray = np.vstack((imgarray,img2[None]))
			labs.append(int(row[1])-48)
		cursor.close()

	labarray = np.array(labs)
	return (imgarray, labarray)

	# test_dataset = tf.data.Dataset.from_tensor_slices((imgarray, labarray))
	# BATCH_SIZE = 32
	# test_dataset = test_dataset.batch(BATCH_SIZE)
	# return test_dataset

def train():
	trainds = getTrainData()
	

	# model = tf.keras.models.Sequential()
	# model.add(tf.keras.layers.Flatten(input_shape=(50, 50,3)));
	# model.add(tf.keras.layers.Dense(128, activation='relu'));
	# model.add(tf.keras.layers.Dropout(0.2));
	# model.add(tf.keras.layers.Dense(10, activation='softmax'));
	# model.compile(keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['sparse_categorical_accuracy'])

	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(64, activation='relu'))
	model.add(tf.keras.layers.Dense(10, activation='softmax'))
	model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['sparse_categorical_accuracy'])

	steps_per_epoch=tf.math.ceil(5002/32).numpy()
	model.fit(trainds, epochs=6,steps_per_epoch=steps_per_epoch)
	model.save('./font_model_5000.h5')

def dumpimg(ds,fn):
	i = ds.flatten().reshape(50,50,3)
	i = i*255.0
	i = i.astype(np.uint8)
	im = Image.fromarray(i)
	im.save('./fonts/'+fn)

def verify():
	imgds,labds = getTestData()
	model = tf.keras.models.load_model('./font_model_5000.h5')
	model.evaluate(imgds,labds, verbose=2)

	for i in range(0,19):
		res = model.predict(imgds[i+20:i+21])
		mxidx = np.argmax(res.flatten())
		fn = 'fnt_'+ str(i)+'_'+str(mxidx)+'.png'
		dumpimg(imgds[i+20],fn)

#train()
verify()
