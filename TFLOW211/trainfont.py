
from PIL import Image
import base64
import io
import numpy as np
import pyodbc

import tensorflow_hub as hub

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import cv2

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

SZ = 224

def convtimg(s64):
	bt = base64.b64decode(s64)
	im = Image.open(io.BytesIO(bt))
	img1 = np.array(im)
	img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
	img1 = cv2.resize(img1,(SZ,SZ))
	img1 = img1.astype(np.float32)
	img1 = img1/255.0
	return img1

#OGP-rect5x1,
# def getTrainData(ogptype,topx):
# 	AUTOTUNE = tf.data.experimental.AUTOTUNE
# 	imgstr = []
# 	labs = []

# 	with pyodbc.connect(Driver='{ODBC Driver 17 for SQL Server}',Server='wuxinpi.chn.ii-vi.net', UID='WATApp', PWD='WATApp@123', Database='WAT') as conn:
# 		cursor = conn.cursor()
# 		sql = "select top 2 [TrainingImg],[ImgVal] from [WAT].[dbo].[AITrainingData] where Revision = '"+ogptype+"'"
# 		cursor.execute(sql)
# 		rows = cursor.fetchall() 
# 		for row in rows:
# 			imgstr.append(str(row[0]))
# 			labs.append(int(row[1])-48)
# 		cursor.close()

# 	img1 = convtimg(imgstr[0])
# 	img2 = convtimg(imgstr[1])
# 	imgarray = np.vstack((img1[None],img2[None]))

# 	#print(img2.shape)
# 	#print(imgarray.shape)

# 	print('start loading data..........')
# 	i = 0
# 	img2list = []
# 	with pyodbc.connect(Driver='{ODBC Driver 17 for SQL Server}',Server='wuxinpi.chn.ii-vi.net', UID='WATApp', PWD='WATApp@123', Database='WAT') as conn:
# 		cursor = conn.cursor()
# 		sql = "select top "+str(topx)+" [TrainingImg],[ImgVal] from [WAT].[dbo].[AITrainingData] where Revision = '"+ogptype+"'" # order by UpdateTime desc"
# 		cursor.execute(sql)
# 		rows = cursor.fetchall() 
# 		for row in rows:
# 			if i%100 == 0:
# 				print("get train data "+str(i)+"..................")
# 			img2 = convtimg(str(row[0]))
# 			img2list.append(img2)
# 			labs.append(int(row[1])-48)
# 			i = i+1
# 		cursor.close()
# 	i = 0
# 	for img2 in img2list:
# 		if i%100 == 0:
# 				print("vstack data "+str(i)+"..................")
# 		imgarray = np.vstack((imgarray,img2[None]))
# 		i = i+1

# 	print("finish data stack................")

# 	labarray = np.array(labs)

# 	label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labarray, tf.int64))
# 	image_ds = tf.data.Dataset.from_tensor_slices(imgarray)
# 	train_dataset = tf.data.Dataset.zip((image_ds, label_ds))

# 	train_dataset = train_dataset.repeat()
# 	BATCH_SIZE = 32
# 	SHUFFLE_BUFFER_SIZE = 500
# 	train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
# 	train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

# 	return train_dataset


def getTrainData(ogptype,topx):
	AUTOTUNE = tf.data.experimental.AUTOTUNE
	imgstr = []
	labs = []

	# with pyodbc.connect(Driver='{ODBC Driver 17 for SQL Server}',Server='wuxinpi.chn.ii-vi.net', UID='WATApp', PWD='WATApp@123', Database='WAT') as conn:
	# 	cursor = conn.cursor()
	# 	sql = "select top 2 [TrainingImg],[ImgVal] from [WAT].[dbo].[AITrainingData] where Revision = '"+ogptype+"'"
	# 	cursor.execute(sql)
	# 	rows = cursor.fetchall() 
	# 	for row in rows:
	# 		imgstr.append(str(row[0]))
	# 		labs.append(int(row[1])-48)
	# 	cursor.close()

	# img1 = convtimg(imgstr[0])
	# img2 = convtimg(imgstr[1])
	# imgarray = np.vstack((img1[None],img2[None]))

	#print(img2.shape)
	#print(imgarray.shape)

	print('start loading data..........')
	i = 0
	img2list = []
	with pyodbc.connect(Driver='{ODBC Driver 17 for SQL Server}',Server='wuxinpi.chn.ii-vi.net', UID='WATApp', PWD='WATApp@123', Database='WAT') as conn:
		cursor = conn.cursor()
		sql = "select top "+str(topx)+" [TrainingImg],[ImgVal] from [WAT].[dbo].[AITrainingData] where Revision = '"+ogptype+"' order by UpdateTime asc"
		cursor.execute(sql)
		rows = cursor.fetchall() 
		for row in rows:
			if i%100 == 0:
				print("get train data "+str(i)+"..................")
			img2 = convtimg(str(row[0]))
			img2list.append(img2)
			labs.append(int(row[1])-48)
			i = i+1
		cursor.close()

	imgarraylist = []
	tempimg = []
	imgarray = None


	i = 0
	for img2 in img2list:
		if i%100 == 0:
				print("vstack data "+str(i)+"..................")

		if i%500 == 0:
			tempimg.append(img2)
			if i != 0:
				imgarraylist.append(imgarray)
			i = i+1
			continue

		if len(tempimg) == 1:
			imgarray = np.vstack((tempimg[0][None],img2[None]))
			tempimg = []
			i = i+1
			continue

		imgarray = np.vstack((imgarray,img2[None]))
		i = i+1

	imgarraylist.append(imgarray)

	print(len(imgarraylist))

	firstarray = None
	i = 0
	for ia in imgarraylist:
		if i == 0:
			firstarray = ia
		else:
			firstarray = np.vstack((firstarray,ia))
		i = i+1

	print("finish data stack................")

	print(firstarray.shape)

	labarray = np.array(labs)

	label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labarray, tf.int64))
	image_ds = tf.data.Dataset.from_tensor_slices(firstarray)
	train_dataset = tf.data.Dataset.zip((image_ds, label_ds))

	train_dataset = train_dataset.repeat()
	BATCH_SIZE = 32
	SHUFFLE_BUFFER_SIZE = 500
	train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
	train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

	return train_dataset

#OGP-rect5x1,
def getTestData(ogptype):
	imgstr = []
	labs = []

	print('start loading data..........')
	i = 0
	img2list = []

	with pyodbc.connect(Driver='{ODBC Driver 17 for SQL Server}',Server='wuxinpi.chn.ii-vi.net', UID='WATApp', PWD='WATApp@123', Database='WAT') as conn:
		cursor = conn.cursor()
		#sql = "select top 150 [TrainingImg],[ImgVal] from [WAT].[dbo].[AITrainingData] where Revision = '"+ogptype+"' order by UpdateTime asc"
		sql = "select top 580 ChildImg,ImgVal from SonImg where MainImgKey in (select MainImgKey from OGPFatherImg where  WaferNum = 'S2102040427')  and ImgVal <> 88 and ImgVal <> 89"
		cursor.execute(sql)
		rows = cursor.fetchall() 
		for row in rows:
			if i%100 == 0:
				print("get train data "+str(i)+"..................")
			img2 = convtimg(str(row[0]))
			img2list.append(img2)
			labs.append(int(row[1])-48)
			i = i+1
		cursor.close()

	imgarraylist = []
	tempimg = []
	imgarray = None


	i = 0
	for img2 in img2list:
		if i%100 == 0:
				print("vstack data "+str(i)+"..................")

		if i%500 == 0:
			tempimg.append(img2)
			if i != 0:
				imgarraylist.append(imgarray)
			i = i+1
			continue

		if len(tempimg) == 1:
			imgarray = np.vstack((tempimg[0][None],img2[None]))
			tempimg = []
			i = i+1
			continue

		imgarray = np.vstack((imgarray,img2[None]))
		i = i+1

	imgarraylist.append(imgarray)

	print(len(imgarraylist))

	firstarray = None
	i = 0
	for ia in imgarraylist:
		if i == 0:
			firstarray = ia
		else:
			firstarray = np.vstack((firstarray,ia))
		i = i+1

	print("finish data stack................")

	labarray = np.array(labs)
	return (firstarray, labarray)

	# test_dataset = tf.data.Dataset.from_tensor_slices((imgarray, labarray))
	# BATCH_SIZE = 32
	# test_dataset = test_dataset.batch(BATCH_SIZE)
	# return test_dataset

def train(ogptype,modelfn,topx,epochs):
	trainds = getTrainData(ogptype,topx)
	
	# model = tf.keras.models.Sequential()
	# model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
	# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
	# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
	# model.add(tf.keras.layers.Flatten())
	# model.add(tf.keras.layers.Dense(64, activation='relu'))
	# #model.add(tf.keras.layers.Dropout(0.2));
	# model.add(tf.keras.layers.Dense(10, activation='softmax'))
	# model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['sparse_categorical_accuracy'])

	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.InputLayer(input_shape=(SZ,SZ,3)))
	model.add(hub.KerasLayer('./hub_model/mbv3',trainable=True))
	model.add(tf.keras.layers.Dropout(rate=0.2))
	model.add(tf.keras.layers.Dense(10, activation='softmax'))
	model.trainable=True

	model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["sparse_categorical_accuracy"])
	model.summary()

	steps_per_epoch=tf.math.ceil((topx+2)/32).numpy()
	model.fit(trainds, epochs=epochs,steps_per_epoch=steps_per_epoch)
	model.save(modelfn)


def dumpimg(ds,fn):
	i = ds.flatten().reshape(224,224,3)
	i = i*255.0
	i = i.astype(np.uint8)
	im = Image.fromarray(i)
	im.save('./fonts/'+fn)

def verify(ogptype,modelfn):
	imgds,labds = getTestData(ogptype)
	model = tf.keras.models.load_model(modelfn, custom_objects={'KerasLayer': hub.KerasLayer})
	model.evaluate(imgds,labds, verbose=2)

	# for i in range(1,2000):
	# 	res = model.predict(imgds[i:i+1])
	# 	mxidx = np.argmax(res.flatten())
	# 	fn = 'fnt_'+ str(i)+'_'+str(mxidx)+'.png'
	# 	dumpimg(imgds[i],fn)

def convertmodeltopb(oldmodelfn,newmodelfn):
	model = tf.keras.models.load_model(oldmodelfn, custom_objects={'KerasLayer': hub.KerasLayer})
	#path of the directory where you want to save your model
	frozen_out_path = './'
	# name of the .pb file
	frozen_graph_filename = newmodelfn
	full_model = tf.function(lambda x: model(x))

	full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

	# Get frozen ConcreteFunction
	frozen_func = convert_variables_to_constants_v2(full_model)
	frozen_func.graph.as_graph_def()
	layers = [op.name for op in frozen_func.graph.get_operations()]

	print("-" * 60)
	print("Frozen model layers: ")
	for layer in layers:
	    print(layer)

	print("-" * 60)
	print("Frozen model inputs: ")
	print(frozen_func.inputs)
	print("Frozen model outputs: ")
	print(frozen_func.outputs)

	# Save frozen graph to disk
	tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
	                  logdir=frozen_out_path,
	                  name=f"{frozen_graph_filename}.pb",
	                  as_text=False)

	# Save its text representation
	# tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
	#                   logdir=frozen_out_path,
	#                   name=f"{frozen_graph_filename}.pbtxt",
	#                   as_text=True)

def verify2(ogptype,newmodelfn):
	imgds,labds = getTestData(ogptype)
	opencv_net = cv2.dnn.readNetFromTensorflow('./'+newmodelfn+'.pb')

	#print("OpenCV model was successfully read. Model layers: \n", opencv_net.getLayerNames())
	for i in range(1,2000):
		img = np.array(imgds[i])
		imgcp = np.copy(img)

		input_blob = cv2.dnn.blobFromImage(
	    image=img,
	    scalefactor=1.0,
	    size=(SZ, SZ),  # img target size
	    mean=(0,0,0),
	    swapRB=False,  # BGR -> RGB
	    crop=False )


		opencv_net.setInput(input_blob)
		out = opencv_net.forward()

		mxidx = np.argmax(out.flatten())
		fn = 'fnt_'+ str(i)+'_'+str(mxidx)+'.png'
		dumpimg(imgcp,fn)

def random_invert_img(x, p=0.5):
	if  tf.random.uniform([]) < p:
		x = (255-x)
	else:
		x
	return x

def data_Enhance():
	IMG_SIZE = 180
	resize_and_rescale = tf.keras.Sequential([
		layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
		layers.experimental.preprocessing.Rescaling(1./255)
	])

	data_augmentation = tf.keras.Sequential([
		layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
		layers.experimental.preprocessing.RandomRotation(0.2),
	])

	random_invert_layer =  layers.Lambda(lambda x: random_invert_img(x, factor))

	model = tf.keras.Sequential([
	resize_and_rescale,
	data_augmentation,
	random_invert_layer,
	layers.Conv2D(16, 3, padding='same', activation='relu'),
	layers.MaxPooling2D()
	])

def img_operate(image):
	flipped = tf.image.flip_left_right(image)
	grayscaled = tf.image.rgb_to_grayscale(image)
	saturated = tf.image.adjust_saturation(image, 3)
	bright = tf.image.adjust_brightness(image, 0.4)
	cropped = tf.image.central_crop(image, central_fraction=0.5)
	rotated = tf.image.rot90(image)


# ogptype = 'OGP-rect5x1'
# modelfn = './font_ogp5x1_8390_6.h5'
# topx = 8390
# epochs=6
# newmodelfn = 'font_ogp5x1_8390_6'

# ogptype = 'OGP-small5x1'
# modelfn = './font_ogpsm5x1_450.h5'
# topx = 450
# epochs= 10
# newmodelfn = 'font_ogpsm5x1_450'

# ogptype = 'OGP-rect2x1'
# modelfn = './font_ogp2x1_5330_5.h5'
# topx = 5330
# epochs= 5
# newmodelfn = 'font_ogp2x1_5330_5'

# ogptype = 'OGP-circle2168'
# modelfn = './font_ogp2168_1160_10.h5'
# topx = 1160
# epochs= 10
# newmodelfn = 'font_ogp2168_1160_10'

ogptype = 'OGP-sm-iivi'
modelfn = './font_ogpsmiivi_1113_14.h5'
topx = 1113
epochs= 14
newmodelfn = 'font_ogpsmiivi_1113_14'

# ogptype = 'OGP-iivi'
# modelfn = './font_ogpiivi_620_16.h5'
# topx = 620
# epochs= 16
# newmodelfn = 'font_ogpiivi_620_16'


# ogptype = 'OGP-A10G'
# modelfn = './font_ogpa10g_750.h5'
# topx = 750
# epochs= 6
# newmodelfn = 'font_ogpa10g_750'

#train(ogptype,modelfn,topx,epochs)

#verify(ogptype,modelfn)

convertmodeltopb(modelfn,newmodelfn)

# verify2(ogptype,newmodelfn)


# modelfn = './font_ogp2x1_5330_8.h5'
# newmodelfn = 'font_ogp2x1_5330_8'
# convertmodeltopb(modelfn,newmodelfn)

# modelfn = './font_ogp5x1_8390_6.h5'
# newmodelfn = 'font_ogp5x1_8390_6'
# convertmodeltopb(modelfn,newmodelfn)

# modelfn = './font_ogpa10g_810_14.h5'
# newmodelfn = 'font_ogpa10g_810_14'
# convertmodeltopb(modelfn,newmodelfn)

# modelfn = './font_ogpsm5x1_503_16.h5'
# newmodelfn = 'font_ogpsm5x1_503_16'
# convertmodeltopb(modelfn,newmodelfn)

# modelfn = './font_ogpsmiivi_1113_6.h5'
# verify(ogptype,modelfn)

# modelfn = './font_ogpsmiivi_1113_8.h5'
# verify(ogptype,modelfn)

# modelfn = './font_ogpsmiivi_1113_10.h5'
# verify(ogptype,modelfn)

# modelfn = './font_ogpsmiivi_1113_12.h5'
# verify(ogptype,modelfn)

# modelfn = './font_ogpsmiivi_1113_14.h5'
# verify(ogptype,modelfn)

# modelfn = './font_ogpsmiivi_1113_16.h5'
# verify(ogptype,modelfn)

