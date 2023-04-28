import os
import io
import pprint
import tempfile
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from six import BytesIO
from IPython import display
from urllib.request import urlopen

import orbit
import tensorflow_models as tfm

from official.core import exp_factory
from official.core import config_definitions as cfg
from official.vision.serving import export_saved_model_lib
from official.vision.ops.preprocess_ops import normalize_image
from official.vision.ops.preprocess_ops import resize_and_crop_image
from official.vision.utils.object_detection import visualization_utils
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder


import dataclasses
import os
from typing import Optional, List, Sequence, Union

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.modeling.hyperparams import base_config
from official.vision.configs import common
from official.vision.configs import decoders
from official.vision.configs import backbones

from official.vision.configs import retinanet

COCO_INPUT_PATH_BASE = 'coco'
COCO_TRAIN_EXAMPLES = 5000
COCO_VAL_EXAMPLES = 500


def retinanet_resnetfpn_coco_local() -> cfg.ExperimentConfig:
  """COCO object detection with RetinaNet."""
  train_batch_size = 256
  eval_batch_size = 8
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=retinanet.RetinaNetTask(
          # init_checkpoint='gs://cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/ckpt-28080',
          # init_checkpoint_modules='backbone',
          # annotation_file=os.path.join(COCO_INPUT_PATH_BASE,
          #                              'instances_val2017.json'),
          model=retinanet.RetinaNet(
              num_classes=91,
              input_size=[640, 640, 3],
              norm_activation=common.NormActivation(use_sync_bn=False),
              min_level=3,
              max_level=7),
          losses=retinanet.Losses(l2_weight_decay=1e-4),
          train_data=retinanet.DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=retinanet.Parser(
                  aug_rand_hflip=True, aug_scale_min=0.8, aug_scale_max=1.2)),
          validation_data=retinanet.DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size)),
      trainer=cfg.TrainerConfig(
          train_steps=72 * steps_per_epoch,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [
                          57 * steps_per_epoch, 67 * steps_per_epoch
                      ],
                      'values': [
                          0.32 * train_batch_size / 256.0,
                          0.032 * train_batch_size / 256.0,
                          0.0032 * train_batch_size / 256.0
                      ],
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 500,
                      'warmup_learning_rate': 0.0067
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config


train_data_input_path = './mydata/train-00000-of-00001.tfrecord'
valid_data_input_path = './mydata/valid-00000-of-00001.tfrecord'
test_data_input_path = './mydata/test-00000-of-00001.tfrecord'
model_dir = './mydata/trained_model/'
export_dir ='./mydata/exported_model/'



#CONFIG PART
exp_config = retinanet_resnetfpn_coco_local() #exp_factory.get_exp_config('retinanet_resnetfpn_coco')
batch_size = 8
num_classes = 3

HEIGHT, WIDTH = 256, 256
IMG_SIZE = [HEIGHT, WIDTH, 3]

# Backbone config.
exp_config.task.freeze_backbone = False
exp_config.task.annotation_file = ''

# Model config.
exp_config.task.model.input_size = IMG_SIZE
exp_config.task.model.num_classes = num_classes + 1
exp_config.task.model.detection_generator.tflite_post_processing.max_classes_per_detection = exp_config.task.model.num_classes

# Training data config.
exp_config.task.train_data.input_path = train_data_input_path
exp_config.task.train_data.dtype = 'float32'
exp_config.task.train_data.global_batch_size = batch_size
exp_config.task.train_data.parser.aug_scale_max = 1.0
exp_config.task.train_data.parser.aug_scale_min = 1.0

# Validation data config.
exp_config.task.validation_data.input_path = valid_data_input_path
exp_config.task.validation_data.dtype = 'float32'
exp_config.task.validation_data.global_batch_size = batch_size


#TRAINER PART
train_steps = 6000
exp_config.trainer.steps_per_loop = 100 # steps_per_loop = num_of_training_examples // train_batch_size

exp_config.trainer.summary_interval = 100
exp_config.trainer.checkpoint_interval = 100
exp_config.trainer.validation_interval = 100
exp_config.trainer.validation_steps =  100 # validation_steps = num_of_validation_examples // eval_batch_size
exp_config.trainer.train_steps = train_steps
exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 100

exp_config.trainer.optimizer_config.learning_rate.type = 'cosine'
exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps
exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.1
exp_config.trainer.optimizer_config.warmup.linear.warmup_learning_rate = 0.05



task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=model_dir)


if exp_config.runtime.mixed_precision_dtype == tf.float16:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')


logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]



if 'GPU' in ''.join(logical_device_names):
	distribution_strategy = tf.distribute.MirroredStrategy()
elif 'TPU' in ''.join(logical_device_names):
	tf.tpu.experimental.initialize_tpu_system()
	tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='/device:TPU_SYSTEM:0')
	distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
	print('Warning: this will be really slow.')
	distribution_strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])



# model, eval_logs = tfm.core.train_lib.run_experiment(
# 	distribution_strategy=distribution_strategy,
#     task=task,
#     mode='train_and_eval',
#     params=exp_config,
#     model_dir=model_dir,
#     run_post_eval=True)



export_saved_model_lib.export_inference_graph(
    input_type='image_tensor',
    batch_size=1,
    input_image_size=[HEIGHT, WIDTH],
    params=exp_config,
    checkpoint_path=tf.train.latest_checkpoint(model_dir),
    export_dir=export_dir)


# imported = tf.saved_model.load(export_dir)
# model_fn = imported.signatures['serving_default']
# result = model_fn(image)