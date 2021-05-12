import tensorflow as tf
from object_detection.utils import dataset_util

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import numpy as np
import cv2
import copy


from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config

def create_tf_example(ih,iw,xmin,xmax,ymin,ymax,fpath,clatx,claid):

    # TODO(user): Populate the following variables from your example.
    height = ih # Image height
    width = iw # Image width
    filename = b'' # Filename of the image. Empty if image is not from file

    encoded_image_data = None
    with tf.io.gfile.GFile(fpath, 'rb') as fid:
        encoded_image_data = fid.read()

    image_format = b'jpeg' # b'jpeg' or b'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmins.append(xmin/iw)
    xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
    xmaxs.append(xmax/iw)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymins.append(ymin/ih)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
    ymaxs.append(ymax/ih)

    classes_text = [] # List of string class name of bounding box (1 per box)
    classes_text.append(bytes(clatx,'utf-8'))
    classes = [] # List of integer class id of bounding box (1 per box)
    classes.append(claid)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def writesiglefile(fn,outfile):
    writer = tf.io.TFRecordWriter(outfile)
    file1 = open(fn,'r')
    lines = file1.readlines()
    for ln in lines:
        sts = ln.strip().split(';')
        tf_example = create_tf_example(int(sts[0]),int(sts[1]),float(sts[2]),float(sts[3]),float(sts[4]),float(sts[5]),sts[6],sts[7],int(sts[8]))
        writer.write(tf_example.SerializeToString())
    writer.close()
    file1.close()


def writemultifile():
    num_shards=10
    output_filebase='./tfrec/eval_dataset.record'

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack, output_filebase, num_shards)
    for index, example in examples:
        tf_example = create_tf_example(example)
        output_shard_index = index % num_shards
        output_tfrecords[output_shard_index].write(tf_example.SerializeToString())


def verifymbv2():
    model = tf.saved_model.load('./tfrec/mbv2frozen/saved_model')

    file1 = open('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\F5X1XY4.txt','r')
    lines = file1.readlines()
    for ln in lines:
        sts = ln.strip().split(';')
        fn = sts[6]
        img = tf.io.read_file(fn)

        image_tensor = tf.io.decode_image(img, channels=3)
        #image_tensor  = tf.image.resize(image_tensor , [640, 640])
        #image_tensor = tf.cast(image_tensor,tf.float32)
        #image_tensor  /= 255.0
        #image_tensor = tf.cast(image_tensor,tf.uint8);
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        output_dict = model(image_tensor)
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy()  for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        # ymin = output_dict['detection_boxes'][0][0]
        # xmin = output_dict['detection_boxes'][0][1]
        # ymax = output_dict['detection_boxes'][0][2]
        # xmax = output_dict['detection_boxes'][0][3]

        cla = output_dict['detection_classes'][0]
        sco = output_dict['detection_scores'][0]

        print(output_dict['detection_boxes'][0])
        print(output_dict['detection_classes'][0])
        print(output_dict['detection_scores'])


        cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
        # srcgray = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(srcgray,(3,3),0)
        # edged = copy.deepcopy(blurred)
        # edged = cv2.Canny(blurred,50,200,edged,3,False)
        high,width,CH = cimg.shape

        drawtangle(output_dict['detection_boxes'][0],cimg,high,width,(0,255,0))
        drawtangle(output_dict['detection_boxes'][1],cimg,high,width,(255,0,0))
        drawtangle(output_dict['detection_boxes'][2],cimg,high,width,(0,0,255))

        f = fn.replace("F5X1-UP","F5X1-UP-VF")
        cv2.imwrite(f,cimg)

def verifymbv2opencv():
    print('try to load model')
    opencv_net = cv2.dnn.readNetFromTensorflow('./tfrec/mbv2frozen2/frozen_inference_graph.pb','./tfrec/mbv2frozen2/frozen_graph.pbtxt')
    print('loaded model')

    file1 = open('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\F5X1XY4.txt','r')
    lines = file1.readlines()
    for ln in lines:
        sts = ln.strip().split(';')
        fn = sts[6]

        img = cv2.imread(fn,cv2.IMREAD_COLOR)
        rows,cols,CH = img.shape
        imgcp = copy.deepcopy(img)
        img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        input_blob = cv2.dnn.blobFromImage(
          image=img2,
          scalefactor=1.0,
          size=(cols, rows),  # img target size
          mean=(0,0,0),
          swapRB=False,  # BGR -> RGB
          crop=False )

        opencv_net.setInput(input_blob)
        networkOutput = opencv_net.forward()

        #print(networkOutput)

        for detection in networkOutput[0,0]:
          score = float(detection[2])
          #if score > 0.015:
          left = detection[3] * cols
          top = detection[4] * rows
          right = detection[5] * cols
          bottom = detection[6] * rows

          #draw a red rectangle around detected objects
          cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)

        f = fn.replace("F5X1-UP","F5X1-UP-VF")
        cv2.imwrite(f,img)

        # num_detections = int(output_dict.pop('num_detections'))
        # output_dict = {key:value[0, :num_detections].numpy()  for key,value in output_dict.items()}
        # output_dict['num_detections'] = num_detections
        # output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        # cla = output_dict['detection_classes'][0]
        # sco = output_dict['detection_scores'][0]

        # print(output_dict['detection_boxes'][0])
        # print(output_dict['detection_classes'][0])
        # print(output_dict['detection_scores'])


        # cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
        # srcgray = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(srcgray,(3,3),0)
        # edged = copy.deepcopy(blurred)
        # edged = cv2.Canny(blurred,50,200,edged,3,False)
        # high,width = edged.shape

        # drawtangle(output_dict['detection_boxes'][0],cimg,high,width,(0,255,0))
        # drawtangle(output_dict['detection_boxes'][1],cimg,high,width,(255,0,0))
        # drawtangle(output_dict['detection_boxes'][2],cimg,high,width,(0,0,255))

        # f = fn.replace("F5X1-UP","F5X1-UP-VF")
        # cv2.imwrite(f,cimg)

def drawtangle(box,cimg,high,width,sc):
    ymin = box[0]
    xmin = box[1]
    ymax = box[2]
    xmax = box[3]
    pxmin = int(xmin*width)
    pymin = int(ymin*high)
    pxmax = int(xmax*width)
    pymax = int(ymax*high)
    cv2.rectangle(cimg,(pxmin,pymin),(pxmax,pymax),sc)


def convertmodeltopb():
  model = tf.saved_model.load('./tfrec/mbv2frozen/saved_model')
  #model = tf.keras.models.load_model(oldmodelfn, custom_objects={'KerasLayer': hub.KerasLayer})
  #path of the directory where you want to save your model
  frozen_out_path = './tfrec/mbv2frozen2'
  # name of the .pb file
  frozen_graph_filename = 'objdetecpb_old'
  full_model = tf.function(lambda x: model(x))

  #full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
  full_model = full_model.get_concrete_function(tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.uint8))
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
  # tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
  #                   logdir=frozen_out_path,
  #                   name=f"{frozen_graph_filename}.pb",
  #                   as_text=False)

  tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pbtxt",
                  as_text=True)

def convertmodeltopb3():
  loaded = tf.saved_model.load('./tfrec/mbv2frozen/saved_model')
  infer = loaded.signatures['serving_default']
  f = tf.function(infer).get_concrete_function(input_1=tf.TensorSpec(shape=[None, 640, 640, 3], dtype=tf.float32))
  f2 = convert_variables_to_constants_v2(f)
  graph_def = f2.graph.as_graph_def()
  # Export frozen graph
  with tf.io.gfile.GFile('./tfrec/mbv2frozen2/objdetecpb.pb', 'wb') as f:
     f.write(graph_def.SerializeToString())

def convertmodeltopb4():
  loaded = tf.saved_model.load('./tfrec/mbv2frozen/saved_model')
  infer = loaded.signatures['serving_default']
  f = tf.function(infer).get_concrete_function(input_tensor=tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.uint8))
  f2 = convert_variables_to_constants_v2(f)
  graph_def = f2.graph.as_graph_def()
  # Export frozen graph
  with tf.io.gfile.GFile('./tfrec/mbv2frozen2/objdetecpbf32.pb', 'wb') as f:
     f.write(graph_def.SerializeToString())

def convertmodeltopb2():
  from tensorflow.python.tools import freeze_graph
  output_node_names = ['StatefulPartitionedCall']
  output_node_names = ','.join(output_node_names)
  save_pb_model_path = './tfrec/mbv2frozen2/objdetecpb.pb'
  input_saved_model_dir='./tfrec/mbv2frozen/saved_model' 

  freeze_graph.freeze_graph(input_graph=None, input_saver=None,
                              input_binary=None,
                              input_checkpoint=None,
                              output_node_names=output_node_names,
                              restore_op_name=None,
                              filename_tensor_name=None,
                              output_graph=save_pb_model_path,
                              clear_devices=None,
                              initializer_nodes=None,
                              input_saved_model_dir=input_saved_model_dir)


#convertmodeltopb()

#verifymbv2opencv()

#convertmodeltopb2()

#convertmodeltopb()

#verifymbv2()

writesiglefile('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\F5X1XY.txt','./tfrec/train_dataset.tfrecord')
writesiglefile('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\F5X1XY2.txt','./tfrec/eval_dataset2.tfrecord')


#TF2.4

#TRAIN COMMAND
# python.exe object_detection/model_main_tf2.py 
# --pipeline_config_path=D:\PlanningForCast\condaenv\tf_models_xxx\research\object_detection\configs\tf2\ssd_mv2_pipeline.config 
# --model_dir=D:\PlanningForCast\condaenv\tfrec\mbv3model --num_train_steps=500 --checkpoint_every_n=50  --alsologtostderr


#python object_detection/exporter_main_v2.py --input_type=image_tensor  --pipeline_config_path=D:\PlanningForCast\condaenv\tf_models_xxx\research\object_detection\configs\tf2\ssd_mv2_pipeline.config 
# --trained_checkpoint_dir=D:\PlanningForCast\condaenv\tfrec\mbv2model 
# --output_directory=D:\PlanningForCast\condaenv\tfrec\mbv2frozen 



#python.exe object_detection/model_main_tf2.py 
#--pipeline_config_path=D:\PlanningForCast\condaenv\tf_models_xxx\research\object_detection\configs\tf2\ssd_mv2_pipeline.config 
# --model_dir=./mv2/saved_model --checkpoint_dir=D:\PlanningForCast\condaenv\tfrec\mbv3model   --alsologtostderr



#TF1.5

#(tflow114) D:\PlanningForCast\condaenv\tf_models_xxx\research>python.exe object_detection/model_main.py --pipeline_config_pat
#h=D:\PlanningForCast\condaenv\tfrec\ssd_mbv2_tf1\pipeline.config --model_dir=D:\PlanningForCast\condaenv\tfrec\mbv2model --nu
#m_train_steps=100



#model.ckpt.data-00000-of-00001
#model.ckpt.index
#model.ckpt.meta

#(tflow114) D:\PlanningForCast\condaenv\tf_models_xxx\research>python object_detection/export_inference_graph.py --input_type=
#image_tensor  --pipeline_config_path=D:/PlanningForCast/condaenv/tfrec/ssd_mbv2_tf1/pipeline.config --trained_checkpoint_pref
#ix=./mbv2model/model.ckpt  --output_directory=./mbv2frozen


#(tflow24) D:\PlanningForCast\condaenv>python.exe ./tfrec/opencv-master/samples/dnn/tf_text_graph_ssd.py --input=./tfrec/mbv2frozen2/frozen_inference_graph.pb --
#output=./tfrec/txt/frozen_grap.pbtxt --config=D:/PlanningForCast/condaenv/tfrec/ssd_mbv2_tf1/pipeline.config

#modify frozen_grap.pbtxt  name: "FeatureExtractor/MobilenetV2/Conv/Conv2D"  input: "image_tensor"

# node {
#   name: "FeatureExtractor/MobilenetV2/Conv/Conv2D"
#   op: "Conv2D"
#   input: "image_tensor"
#   input: "FeatureExtractor/MobilenetV2/Conv/weights"
#   attr {
#     key: "data_format"
#     value {
#       s: "NHWC"
#     }
#   }