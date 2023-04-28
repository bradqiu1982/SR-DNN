

import os
import sys
# import warnings
# warnings.filterwarnings("ignore")

import logging
logging.getLogger('absl').setLevel('ERROR')

import tensorflow as tf
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import io
# import matplotlib
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)

import pathlib
import cv2
import copy
import uuid

import math

import colorsys
import random

from PIL import Image
# from six import BytesIO

# import orbit
# import tensorflow_models as tfm


# from official.core import exp_factory
# from official.core import config_definitions as cfg
# from official.vision.serving import export_saved_model_lib
# from official.vision.ops.preprocess_ops import normalize_image
# from official.vision.ops.preprocess_ops import resize_image
# from official.vision.utils.object_detection import visualization_utils
# from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder
# from official.vision.ops.preprocess_ops 

from absl import app
from absl import flags
from absl import logging
# import gin



# from official.common import distribute_utils
# from official.common import flags as tfm_flags
# from official.core import task_factory
# from official.core import train_lib
# from official.core import train_utils
# from official.modeling import performance
# from official.vision import registry_imports  # pylint: disable=unused-import
# from official.vision.utils import summary_manager


import dataclasses
from typing import Optional, List, Sequence, Union

# from official.core import config_definitions as cfg
# from official.core import exp_factory
# from official.modeling import hyperparams
# from official.modeling import optimization
# from official.modeling.hyperparams import base_config
# from official.vision.configs import common
# from official.vision.configs import decoders
# from official.vision.configs import backbones

# from official.vision.configs import retinanet


def random_colors(N, bright=True):
	brightness = 1.0 if bright else 0.7
	hsv = [(i / N, 1, brightness) for i in range(N)]
	colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
	random.shuffle(colors)
	return colors


def drawtangle(box,cls,maskts,cimg,color):
	ymin = int(box[0])
	xmin = int(box[1])
	ymax = int(box[2])
	xmax = int(box[3])

	cx = int((xmin+xmax)/2)
	cy = int((ymin+ymax)/2)

	cv2.rectangle(cimg,(xmin,ymin),(xmax,ymax),(0,255,0),2)
	


	mask = maskts.numpy()
	mask = cv2.resize(mask, ((xmax-xmin), (ymax-ymin)))
	mask = (mask*255).astype("uint8")
	# print('mask:')
	# print(mask)


	rnnmask = np.zeros((640,640), dtype="uint8")
	rnnmask[ymin:ymax,xmin:xmax] = mask

	alpha=0.5
	for c in range(3):
		cimg[:, :, c] = np.where(rnnmask > 50,cimg[:, :, c] *(1 - alpha) + alpha * color[c] * 255,cimg[:, :, c])

	cv2.putText(cimg,str(int(cls)),(xmin+10,ymin+20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),3)

	# im = Image.fromarray(rnnmask)
	# im.save('./mydata/mask'+str(i)+'.jpg')


def GetActualImageBond(fn):
	cimg = cv2.imread(fn,cv2.IMREAD_GRAYSCALE)
	blurred = cv2.GaussianBlur(cimg,(3,3),0)
	edges = cv2.Canny(blurred,30,180)
	hg,wd = cimg.shape

	ckleft = -1
	for x in range(20,int(wd/2),1):
		submat = edges[20:hg-20,x:x+2]
		nonzero = cv2.countNonZero(submat)
		if nonzero > 100:
			ckleft = x
			break

	ckright = -1
	for x in range(wd-20,int(wd/2),-1):
		submat = edges[20:hg-20,x-2:x]
		nonzero = cv2.countNonZero(submat)
		if nonzero > 100:
			ckright = x
			break



	cktop = -1
	for y in range(0,int(hg/2),1):
		submat = edges[y:y+2,20:wd-20]
		nonzero = cv2.countNonZero(submat)
		if nonzero > 100:
			cktop = y
			break


	ckbotm = -1
	for y in range(hg,int(hg/2),-1):
		submat = edges[y-2:y,20:wd-20]
		nonzero = cv2.countNonZero(submat)
		if nonzero > 100:
			ckbotm = y
			break

	return ckleft,ckright,cktop,ckbotm


def GenerateImg4AOI(fn,newpath):
	ckleft,ckright,cktop,ckbotm = GetActualImageBond(fn)
	if ckleft != -1 and ckright != -1 and cktop != -1 and ckbotm != -1:
		filename = os.path.basename(fn)
		cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
		cimg = cimg[cktop:ckbotm,ckleft:ckright]
		hg,wd,ch = cimg.shape
		WIDTH = 400 #int((480.0/float(hg))*wd)
		cimg = cv2.resize(cimg,(WIDTH,480))
		array_created = np.full((480, 640, 3),0, dtype = np.uint8)
		start = 0 #int((640-WIDTH)/2)
		array_created[0:480,start:start+WIDTH] = cimg
		grayimg = cv2.cvtColor(array_created,cv2.COLOR_BGR2GRAY)
		cv2.imwrite(newpath+'/'+ filename,grayimg,[cv2.IMWRITE_JPEG_QUALITY,100])
		return newpath+'/'+ filename
	return fn



HIGH = 640
WIDTH = 640
export_dir = './mydata/exported_model_1600_mz_gray101'
imported = tf.saved_model.load(export_dir)
model_fn = imported.signatures['serving_default']
colors = random_colors(10)
input_image_size = (HIGH, WIDTH)
score = 0.9

data_root = pathlib.Path('./mydata/check_piczu')
all_image_paths = list(data_root.glob('*'))
all_image_paths = [str(path) for path in all_image_paths]
for fn in all_image_paths:
	if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
		
		realfn = fn
		cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
		hg,wd,ch = cimg.shape
		if hg != 480 or wd != 640:
			realfn = GenerateImg4AOI(fn,'./mydata/tempdir')
		else:
			img = cv2.imread(fn,cv2.IMREAD_COLOR)
			grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			realfn = './mydata/tempdir/'+os.path.basename(fn).upper().replace('.BMP','.JPG').replace('.PNG','.JPG')
			cv2.imwrite(realfn,grayimg,[cv2.IMWRITE_JPEG_QUALITY,100])

		img = tf.io.read_file(realfn)
		img_tensor = tf.io.decode_image(img, channels=3)
		img_tensor = tf.image.resize(img_tensor,input_image_size)
		img_tensor = tf.expand_dims(img_tensor, axis=0)
		img_tensor = tf.cast(img_tensor, dtype = tf.uint8)

		output_dict = model_fn(img_tensor)
		print(output_dict)
		
		cimg = cv2.imread(realfn,cv2.IMREAD_COLOR)
		cimg = cv2.resize(cimg,(WIDTH,HIGH))
		for i in range(100):
			if float(output_dict['detection_scores'][0][i]) >= score:
				color = colors[i%10]
				drawtangle(output_dict['detection_boxes'][0][i],output_dict['detection_classes'][0][i],output_dict['detection_masks'][0][i],cimg,color)

		# cimg = cv2.resize(cimg,(640,480))
		cv2.imwrite(realfn.replace('.jp','_2000_'+str(int(score*100.0))+'.jp').replace('check_pic','checked_pic').replace('tempdir','checked_piczu'),cimg)
