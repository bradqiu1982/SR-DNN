

import os

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


import pathlib
import cv2
import copy
import uuid

import math

# from PIL import Image
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

HIGH = 480
WIDTH = 480

label_names = ['A10','F2X1','F5X1','SIXINCH','ZURICH']

# export_dir = './mydata/exported_model_retina_base_1000_spw'
export_dir = './mydata/exported_model_imgcls_2000'


imported = tf.saved_model.load(export_dir)
model_fn = imported.signatures['serving_default']


input_image_size = (HIGH, WIDTH)
score = 0.85

data_root = pathlib.Path('./mydata/check_img_cls')
all_image_paths = list(data_root.glob('*'))
all_image_paths = [str(path) for path in all_image_paths]
for fn in all_image_paths:
	if ('.JPG' in fn.upper() or '.JPEG' in fn.upper() or '.PNG' in fn.upper() or '.BMP' in fn.upper()):
		img = tf.io.read_file(fn)
		img_tensor = tf.io.decode_image(img, channels=3)
		img_tensor = tf.image.resize(img_tensor,input_image_size)
		img_tensor = tf.expand_dims(img_tensor, axis=0)
		img_tensor = tf.cast(img_tensor, dtype = tf.uint8)

		output_dict = model_fn(img_tensor)
		print(output_dict)
		index = (tf.argmax(output_dict['logits'], axis=1)[0]).numpy()
		prob = (output_dict['probs'][0][index]).numpy()
		
		cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
		cimg = cv2.resize(cimg,(WIDTH,HIGH))

		lb = label_names[index] + '  {:.2%}'.format(prob)
		cv2.putText(cimg,lb,(50,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
		cv2.imwrite(fn.replace('check_img','checked_img'),cimg)



