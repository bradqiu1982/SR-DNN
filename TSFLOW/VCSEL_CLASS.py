
import tensorflow as tf
import pathlib
import random
import matplotlib.pyplot as plt
import os
from PIL import Image

import tensorflow_hub as hub

import numpy as np
from tensorflow import keras

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import cv2
import copy
import uuid

SZ=454
#SZ = 224
#FILEFOLDER = '\\\\wux-engsys01\\PlanningForCast\\VMI'
FILEFOLDER = '\\\\wux-engsys01\\PlanningForCast\\VCSEL4'
#FILEFOLDER = '\\\\wux-engsys01\\PlanningForCast\\flowers'
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
#os.environ["TFHUB_CACHE_DIR"] = "./hub_model"

#label_names = ['A10','F2X1','F5X1','II_VI','SixInch']

label_names = ['A10-UP','A10-RT','A10-DW','A10-LF'
			,'F2X1-UP','F2X1-RT','F2X1-DW','F2X1-LF'
			,'F5X1-UP','F5X1-RT','F5X1-DW','F5X1-LF'
			,'IIVI-UP','IIVI-RT','IIVI-DW','IIVI-LF'
			,'SIX-UP','SIX-RT','SIX-DW','SIX-LF']

WD=640
HT=480

def preprocess_image(image_raw):
	img_tensor  = tf.image.decode_png(image_raw, channels=3)
	img_tensor  = tf.image.resize(img_tensor ,[SZ,SZ]) #[HT,WD])
	img_tensor  /= 255.0  # normalize to [0,1] range
	return img_tensor

def load_and_preprocess_image(path):
	image = tf.io.read_file(path)
	return preprocess_image(image)


# def change_range(image,label):
# 		return 2*image-1, label

def get_training_ds():
	data_root = pathlib.Path(FILEFOLDER)

	all_image_paths = list(data_root.glob('*/*'))
	all_image_paths = [str(path) for path in all_image_paths]
	random.shuffle(all_image_paths)
	#label_names = ['A10','F2X1','F5X1','II_VI','SixInch'] #sorted(item.name for item in data_root.glob('*/') if item.is_dir())
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
	model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(HT, WD, 3)))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Dropout(0.2));
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dense(classcnt, activation='softmax'))

	#model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['sparse_categorical_accuracy'])
	model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['sparse_categorical_accuracy'])

	steps_per_epoch=tf.math.ceil((image_count+2)/BATCH_SIZE).numpy()
	model.fit(ds, epochs=6,steps_per_epoch=steps_per_epoch)
	model.save('./VCSEL_CLASS_self.h5')


def train_by_mobilev3_hub():
	ds,image_count,classcnt = get_training_ds()
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.InputLayer(input_shape=(SZ,SZ,3)))
	model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(3, (3, 3), activation='relu'))
	#model.add(hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5", trainable=True))
	model.add(hub.KerasLayer('./hub_model/mbv3',trainable=True))
	model.add(tf.keras.layers.Dropout(rate=0.2))
	model.add(tf.keras.layers.Dense(classcnt, activation='softmax'))
	model.trainable=True

	model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["sparse_categorical_accuracy"])
	model.summary()

	#steps_per_epoch=tf.math.ceil(image_count/BATCH_SIZE).numpy()
	model.fit(ds,epochs=4,steps_per_epoch=160)
	model.save('./VCSEL_CLASS_mobilev3_dir_new4.h5')

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

def VerifyVCSEL():
	model = tf.keras.models.load_model('./VCSEL_CLASS_mobilev3_temp.h5', custom_objects={'KerasLayer': hub.KerasLayer})
	#model = tf.keras.models.load_model('./VCSEL_CLASS_self_71_6_77.h5')
	print('loaded model')

	#label_names = ['A10','F2X1','F5X1','II_VI','SixInch'] #sorted(item.name for item in data_root.glob('*/') if item.is_dir())
	label_to_index = dict((index,name) for index, name in enumerate(label_names))

	data_root = pathlib.Path('\\\\wux-engsys01\\PlanningForCast\\VCSEL_VERIFY\\v1')
	all_image_paths = list(data_root.glob('*'))
	all_image_paths = [str(path) for path in all_image_paths]

	#print(all_image_paths)
	for fn in all_image_paths:
		print(fn)
		img = tf.io.read_file(fn)
		image_tensor = tf.io.decode_image(img, channels=3)
		image_tensor  = tf.image.resize(image_tensor ,[SZ,SZ]) #[HT, WD])
		image_tensor  /= 255.0
		#image_tensor = 2*image_tensor - 1
		image_tensor = tf.expand_dims(image_tensor, axis=0)
		res = model.predict(image_tensor)
		mxidx = np.argmax(res.flatten())
		print(mxidx)
		lb = label_to_index[mxidx]
		print(lb)
		print('{:.0f}'.format(100 * np.max(res.flatten())))

		# res = model(image_tensor)
		# mxidx = np.argmax(res.numpy().flatten())
		# print(mxidx)
		# lb = label_to_index[mxidx]
		# print(lb)
		# print('{:.0f}'.format(100 * np.max(res.numpy().flatten())))

		f = fn.replace('.jpg',lb+'.jpg')
		tf.io.write_file(f,img)

def VerifyVCSELwithOpenCV():
	print('try to load model')
	opencv_net = cv2.dnn.readNetFromTensorflow('./VCSEL_CLASS_mobilev3_dir_new4.pb')
	print('loaded model')

	#label_names = ['A10','F2X1','F5X1','II_VI','SixInch'] #sorted(item.name for item in data_root.glob('*/') if item.is_dir())
	label_to_index = dict((index,name) for index, name in enumerate(label_names))

	data_root = pathlib.Path('\\\\wux-engsys01\\PlanningForCast\\VCSEL_VERIFY\\v1')
	all_image_paths = list(data_root.glob('*'))
	all_image_paths = [str(path) for path in all_image_paths]

	#print(all_image_paths)
	for fn in all_image_paths:
		print(fn)
		img = cv2.imread(fn,cv2.IMREAD_COLOR)
		imgcp = copy.deepcopy(img)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		img = cv2.resize(img,(SZ,SZ))
		img = img.astype(np.float32)
		img = img/255.0

		input_blob = cv2.dnn.blobFromImage(
	    image=img,
	    scalefactor=1.0,
	    size=(SZ, SZ),  # img target size
	    mean=(0,0,0),
	    swapRB=False,  # BGR -> RGB
	    crop=False )

		opencv_net.setInput(input_blob)
		res = opencv_net.forward()

		mxidx = np.argmax(res.flatten())
		print(mxidx)
		lb = label_to_index[mxidx]
		print(lb)
		cfd = 100 * np.max(res.flatten())
		print('{:.0f}'.format(cfd))

		f = fn.replace('.jpg','_'+lb+'_'+str(cfd)[0:2]+'.png').replace('.png','_'+lb+'_'+str(cfd)[0:2]+'.png').replace('.bmp','_'+lb+'_'+str(cfd)[0:2]+'.png').replace('.BMP','_'+lb+'_'+str(cfd)[0:2]+'.png').replace('.Bmp','_'+lb+'_'+str(cfd)[0:2]+'.png')

		cv2.imwrite(f,imgcp)

def VerifyModels(tp):
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


	model = ''
	if tp == 'VGG19':
		print('VGG19 MODEL')
		model = tf.keras.models.load_model('./FLOWER_CLASS_VGG19.h5')
	elif tp == 'MBV3':
		print('MOBILEV3 MODEL')
		image_tensor = 2*image_tensor - 1
		model = tf.keras.models.load_model('./FLOWER_CLASS_mobilev3.h5', custom_objects={'KerasLayer': hub.KerasLayer})
	else:
		print('SELF MODEL')
		model = tf.keras.models.load_model('./VCSEL_CLASS_self.h5')

	image_tensor = tf.expand_dims(image_tensor, axis=0)
	res = model.predict(image_tensor)
	print(res.flatten())
	mxidx = np.argmax(res.flatten())
	print(label_to_index[mxidx])

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


def convertimage():
	data_root = pathlib.Path('\\\\wux-engsys01\\PlanningForCast\\VCSEL3\\SixInch-BAK')
	all_image_paths = list(data_root.glob('*'))

	all_image_paths = [str(path) for path in all_image_paths]
	for fn in all_image_paths:
		pt = pathlib.Path(fn).parents[0]

		uid = uuid.uuid4()
		fd = str(pt)+'\\fd'+str(uid)
		img = cv2.imread(fn,cv2.IMREAD_COLOR)

		f0 = fd+'_00.jpg'
		cv2.imwrite(f0,img)

		img = cv2.transpose(img)
		img = cv2.flip(img,1)
		f1 = fd+'_90.jpg'
		cv2.imwrite(f1,img)

		img = cv2.transpose(img)
		img =cv2.flip(img,1)
		f2 =  fd+'_180.jpg'
		cv2.imwrite(f2,img)

		img = cv2.transpose(img)
		img =cv2.flip(img,1)
		f3 = fd+'_270.jpg'
		cv2.imwrite(f3,img)



#train_by_self()
#VerifySelfModel()
#train_by_vgg19()
#train_by_mobilev3_hub()
#'VGG19','MBV3'
#VerifyModels('MBV3')

#convertmodeltopb('./VCSEL_CLASS_mobilev3_dir_new4.h5','VCSEL_CLASS_mobilev3_dir_new4')

#train_by_self()
#train_by_mobilev3_hub()
#convertimage()
#VerifyVCSEL()

#train_by_mobilev3_hub()

VerifyVCSELwithOpenCV()

