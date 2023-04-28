
import os

# import warnings
# warnings.filterwarnings("ignore")

import logging
logging.getLogger('absl').setLevel('ERROR')

import tensorflow as tf
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import io
import matplotlib
import numpy as np


import pathlib
import cv2
import copy
import uuid

import math

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


from absl import app
from absl import flags
from absl import logging
import gin



# from official.common import distribute_utils
# from official.common import flags as tfm_flags
# from official.core import task_factory
# from official.core import train_lib
# from official.core import train_utils
# from official.modeling import performance
# from official.vision import registry_imports  # pylint: disable=unused-import


# import dataclasses
# from typing import Optional, List, Sequence, Union

# from official.core import config_definitions as cfg
# from official.core import exp_factory
# from official.modeling import hyperparams
# from official.modeling import optimization
# from official.modeling.hyperparams import base_config
# from official.vision.configs import common
# from official.vision.configs import decoders
# from official.vision.configs import backbones

import json
from flask import *

from multiprocessing import Lock



zurichlock = Lock()
wblock = Lock()
cache = dict()


# initialize the flask application
app = Flask(__name__)

def  getCOCOBJDectModel():
	COCMODEL='COCMODEL'
	if COCMODEL not in cache:
		model = tf.saved_model.load('./ObjectDetectModel/COCLASER/FASTRCNN')
		# model = tf.saved_model.load('./ObjectDetectModel/COCLASER/RETINANET')
		cache[COCMODEL] = model
		return model
	else:
		return cache[COCMODEL]

@app.route("/COCLASERWBDETECT", methods=["POST"])
def COCLASERWBDETECT():
	HIGH = 640
	WIDTH = 640
	input_image_size = (HIGH, WIDTH)
	score = 0.4

	try:
		res = []
		request_json = request.get_json()
		#print(request_json['imgpath'])

		imported =  getCOCOBJDectModel()
		model_fn = imported.signatures['serving_default']

		idx = 0
		data_root = pathlib.Path(request_json['imgpath'])
		all_image_paths = list(data_root.glob('*'))
		fs = [str(path) for path in all_image_paths]
		for f in fs:
			if '.JPG' in f.upper() or '.JPEG' in f.upper() or '.PNG' in f.upper() or '.BMP' in f.upper():
				print(f)

				wblock.acquire()

				try:
					img = tf.io.read_file(f)
					img_tensor = tf.io.decode_image(img, channels=3)
					img_tensor = tf.image.resize(img_tensor,input_image_size)
					img_tensor = tf.expand_dims(img_tensor, axis=0)
					img_tensor = tf.cast(img_tensor, dtype = tf.uint8)

					output_dict = model_fn(img_tensor)
					for i in range(100):
						if float(output_dict['detection_scores'][0][i]) >= score:
							item = {}
							item['imgname'] = f
							item['score'] = str(float(output_dict['detection_scores'][0][i]))
							item['classid'] = str(int(output_dict['detection_classes'][0][i]))
							box = output_dict['detection_boxes'][0][i]
							item['top'] = str(float(box[0]))
							item['left'] = str(float(box[1]))
							item['botm'] = str(float(box[2]))
							item['right'] = str(float(box[3]))
							res.append(item)
				except:
					exception_message = sys.exc_info()[1]
					print(str(exception_message))
				finally:
					wblock.release()

				if idx > 9:
					break
				idx = idx + 1

		response = jsonify(res)
		response.status_code = 200
	except:
		exception_message = sys.exc_info()[1]
		response = jsonify({"content":str(exception_message)})
		response.status_code = 400

	return response


def  getZURICHOBJDectModel():
	COCMODEL='ZURICHMODEL'
	if COCMODEL not in cache:
		model = tf.saved_model.load('./ObjectDetectModel/AOI/ZURICHVCSEL')
		cache[COCMODEL] = model
		return model
	else:
		return cache[COCMODEL]

@app.route("/ZURICHVCSELAOI", methods=["POST"])
def ZURICHVCSELAOI():
	HIGH = 640
	WIDTH = 640
	input_image_size = (HIGH, WIDTH)
	score = 0.8

	try:
		res = []
		request_json = request.get_json()
		#print(request_json['imgpath'])

		imported =  getZURICHOBJDectModel()
		model_fn = imported.signatures['serving_default']

		idx = 0
		data_root = pathlib.Path(request_json['imgpath'])
		all_image_paths = list(data_root.glob('*'))
		fs = [str(path) for path in all_image_paths]
		for f in fs:
			if '.JPG' in f.upper() or '.JPEG' in f.upper() or '.PNG' in f.upper() or '.BMP' in f.upper():
				print(f)

				zurichlock.acquire()

				try:
					img = tf.io.read_file(f)
					img_tensor = tf.io.decode_image(img, channels=3)
					img_tensor = tf.image.resize(img_tensor,input_image_size)
					img_tensor = tf.expand_dims(img_tensor, axis=0)
					img_tensor = tf.cast(img_tensor, dtype = tf.uint8)

					output_dict = model_fn(img_tensor)
					for i in range(100):
						if float(output_dict['detection_scores'][0][i]) >= score:
							item = {}
							item['imgname'] = f
							item['score'] = str(float(output_dict['detection_scores'][0][i]))
							item['classid'] = str(int(output_dict['detection_classes'][0][i]))
							box = output_dict['detection_boxes'][0][i]
							item['top'] = str(float(box[0]))
							item['left'] = str(float(box[1]))
							item['botm'] = str(float(box[2]))
							item['right'] = str(float(box[3]))

							ymin = int(box[0])
							xmin = int(box[1])
							ymax = int(box[2])
							xmax = int(box[3])
							mask = (output_dict['detection_masks'][0][i]).numpy()
							mask = cv2.resize(mask, ((xmax-xmin), (ymax-ymin)))
							mask = (mask*255).astype("uint8")
							rnnmask = np.zeros((640,640), dtype="uint8")
							rnnmask[ymin:ymax,xmin:xmax] = mask

							# im = Image.fromarray(rnnmask)
							# im.save('./mask'+str(i)+'.jpg')

							item['mask'] = rnnmask.flatten().tolist()

							res.append(item)
				except:
					exception_message = sys.exc_info()[1]
					print(str(exception_message))
				finally:
					zurichlock.release()

				if idx > 9:
					break
				idx = idx + 1

		response = jsonify(res)
		response.status_code = 200
	except:
		exception_message = sys.exc_info()[1]
		response = jsonify({"content":str(exception_message)})
		response.status_code = 400

	return response




if __name__ == "__main__":
	from waitress import serve
	serve(app, host="0.0.0.0", port=5001)
#    run flask application in debug mode
	# app.run(debug=True)
