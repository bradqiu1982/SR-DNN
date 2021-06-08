import tensorflow as tf
from object_detection.utils import dataset_util

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import numpy as np
import cv2
import copy

import pathlib

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config

from tensorflow.python.client import session

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

def load_graph(filename):
    # f = open(filename, 'rb')
    # graph_def = tf.Graph().as_graph_def()
    # graph_def.ParseFromString(f.read())
    # with tf.Graph().as_default() as graph:
    #     tf.import_graph_def(graph_def, name="")
    # # layers = [op.name for op in graph.get_operations()]
    # # for layer in layers:
    # #     print(layer, file=open("./output.txt", "a"))

    # return graph

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def run_inference_for_single_image(image, graph):
    with session.Session(graph=graph) as sess:
        # Get handles to input and output tensors
        ops = graph.get_operations()
        all_tensor_names = {
            output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = graph.get_tensor_by_name(tensor_name)
       
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        # Run inference
        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict

def load_graph_tf1(filename):
    graph = tf.Graph()
    with tf.gfile.GFile(model_filepath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read()) 
    graph.finalize()
    return graph


def verifymbv2pb2():
    tf.enable_eager_execution()
    graph = load_graph('./tfrec/faster_resnet101/frozen_inference_graph.pb')
    fn = './tfrec/dog.jpg'
    img = tf.io.read_file(fn)
    image_tensor = tf.io.decode_image(img, channels=3)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    output_dict = run_inference_for_single_image(image_tensor.numpy(),graph)
    cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
    high,width,CH = cimg.shape
    drawtangle(output_dict['detection_boxes'][0],cimg,high,width,(0,255,0))
    cv2.imwrite('./tfrec/dog2.jpg',cimg)



def verifymbv2pb3():
    tf.enable_eager_execution()

    graph = load_graph('./tfrec/fast_model_frozen/frozen_inference_graph.pb')

    fn = './tfrec/vcsel.jpg'
    img = tf.io.read_file(fn)
    image_tensor = tf.io.decode_image(img, channels=3)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    output_dict = run_inference_for_single_image(image_tensor.numpy(),graph)
    cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
    high,width,CH = cimg.shape
    drawtangle(output_dict['detection_boxes'][0],cimg,high,width,(0,255,0))
    cv2.imwrite('./tfrec/vcsel2.jpg',cimg)

def verifymbv2pb4():
    model = tf.saved_model.load('./tfrec/fast_model_frozen/saved_model')
    #detector = model.signatures['serving_default']
    file1 = open('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\F5X1XY4.txt','r')
    lines = file1.readlines()
    for ln in lines:
        sts = ln.strip().split(';')
        fn = sts[6]
        img = tf.io.read_file(fn)
        image_tensor = tf.io.decode_image(img, channels=3)
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        output_dict = model(image_tensor)

        # print(output_dict, file=open("./res.txt", "a"))
        cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
        high,width,CH = cimg.shape
        drawtangle(output_dict['detection_boxes'][0][0],cimg,high,width,(0,0,255))
        f = fn.replace("F5X1-UP","F5X1-UP-VF")
        cv2.imwrite(f,cimg)

def verifymbv2pb5():
    model = tf.saved_model.load('./tfrec/fast_model_frozen/saved_model')
    fn = './tfrec/vcsel.jpg'
    img = tf.io.read_file(fn)
    image_tensor = tf.io.decode_image(img, channels=3)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    output_dict = model(image_tensor)

    # print(output_dict, file=open("./res.txt", "a"))
    cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
    high,width,CH = cimg.shape
    drawtangle(output_dict['detection_boxes'][0][0],cimg,high,width,(0,0,255))
    cv2.imwrite('./tfrec/vcsel2.jpg',cimg)


def verifymbv2pb7():
    model = tf.saved_model.load('./tfrec/six_faster_savedmodel/saved_model')
    fn = './tfrec/vcselsix.jpg'
    img = tf.io.read_file(fn)
    image_tensor = tf.io.decode_image(img, channels=3)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    output_dict = model(image_tensor)

    # print(output_dict, file=open("./res.txt", "a"))
    cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
    high,width,CH = cimg.shape
    drawtangle(output_dict['detection_boxes'][0][0],cimg,high,width,(0,0,255))
    cv2.imwrite('./tfrec/vcselsix2.jpg',cimg)

def verifymbv2pb8():
    model = tf.saved_model.load('./tfrec/iivi_faster_savedmodel/saved_model')
    fn = './tfrec/vcseliivi.jpg'
    img = tf.io.read_file(fn)
    image_tensor = tf.io.decode_image(img, channels=3)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    output_dict = model(image_tensor)

    # print(output_dict, file=open("./res.txt", "a"))
    cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
    high,width,CH = cimg.shape
    drawtangle(output_dict['detection_boxes'][0][0],cimg,high,width,(0,0,255))
    drawtangle(output_dict['detection_boxes'][0][1],cimg,high,width,(0,0,255))
    cv2.imwrite('./tfrec/vcseliivi2.jpg',cimg)

def verifymbv2pb9():
    model = tf.saved_model.load('./tfrec/iivi_faster_savedmodel/saved_model')
    data_root = pathlib.Path('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\IIVI-VF')
    all_image_paths = list(data_root.glob('*'))
    fs = [str(path) for path in all_image_paths]
    for fn in fs:
        if '.JPG' in fn.upper() or '.JPEG' in fn.upper():
            img = tf.io.read_file(fn)
            image_tensor = tf.io.decode_image(img, channels=3)
            image_tensor = tf.expand_dims(image_tensor, axis=0)
            output_dict = model(image_tensor)

            # print(output_dict, file=open("./res.txt", "a"))
            cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
            high,width,CH = cimg.shape
            drawtangle(output_dict['detection_boxes'][0][0],cimg,high,width,(0,0,255))
            drawtangle(output_dict['detection_boxes'][0][1],cimg,high,width,(0,0,255))
            cv2.imwrite(fn.replace('-VF','-VF2'),cimg)

def verifymbv2pb6():
    configs = config_util.get_configs_from_pipeline_file('./tfrec/fast_model_frozen/pipeline.config')
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore('./tfrec/fast_model_frozen/checkpoint/ckpt-0').expect_partial()


    fn = './tfrec/vcsel.jpg'
    img = tf.io.read_file(fn)
    image_tensor = tf.io.decode_image(img, channels=3)
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    img, shapes = detection_model.preprocess(image_tensor)
    prediction_dict = detection_model.predict(img, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    # print(output_dict, file=open("./res.txt", "a"))
    cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
    high,width,CH = cimg.shape
    drawtangle(output_dict['detection_boxes'][0][0],cimg,high,width,(0,0,255))
    cv2.imwrite('./tfrec/vcsel2.jpg',cimg)


def verifymbv2pb():
    tf.enable_eager_execution()
    graph = load_graph('./tfrec/fast_model_frozen/frozen_inference_graph.pb')

    file1 = open('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\F5X1XY4.txt','r')
    lines = file1.readlines()
    for ln in lines:
        sts = ln.strip().split(';')
        fn = sts[6]
        img = tf.io.read_file(fn)
        image_tensor = tf.io.decode_image(img, channels=3)
        #image_tensor  = tf.image.resize(image_tensor ,[300,300])
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        outputdict = run_inference_for_single_image(image_tensor.numpy(),graph)
        print(outputdict, file=open("./res.txt", "a"))

def verifymbv2pb_old():
    graph = load_graph('./tfrec/mbv2frozen2-5000step/frozen_inference_graph.pb')
    x = graph.get_tensor_by_name('image_tensor:0')
    y1 = graph.get_tensor_by_name('raw_detection_scores:0')
    y2 = graph.get_tensor_by_name('raw_detection_boxes:0')
    y3 = graph.get_tensor_by_name('detection_classes:0')

    file1 = open('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\F5X1XY4.txt','r')
    lines = file1.readlines()
    for ln in lines:
        sts = ln.strip().split(';')
        fn = sts[6]
        img = tf.io.read_file(fn)
        image_tensor = tf.io.decode_image(img, channels=3)
        image_tensor  = tf.image.resize(image_tensor ,[300,300])
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        with session.Session(graph=graph) as sess:
            scores = sess.run(y1, feed_dict={x: image_tensor.numpy()})
            print(scores)
            boxes = sess.run(y2, feed_dict={x: image_tensor.numpy()})
            print(boxes)
            clas = sess.run(y3, feed_dict={x: image_tensor.numpy()})
            print(clas)


def verifymbv2():
    model = tf.saved_model.load('./tfrec/mbv1frozen/saved_model')

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

        print(outputdict)

        # num_detections = int(output_dict.pop('num_detections'))
        # output_dict = {key:value[0, :num_detections].numpy()  for key,value in output_dict.items()}
        # output_dict['num_detections'] = num_detections
        # output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        # # ymin = output_dict['detection_boxes'][0][0]
        # # xmin = output_dict['detection_boxes'][0][1]
        # # ymax = output_dict['detection_boxes'][0][2]
        # # xmax = output_dict['detection_boxes'][0][3]

        # cla = output_dict['detection_classes'][0]
        # sco = output_dict['detection_scores'][0]

        # print(output_dict['detection_boxes'][0])
        # print(output_dict['detection_classes'][0])
        # print(output_dict['detection_scores'])


        # cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
        # # srcgray = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)
        # # blurred = cv2.GaussianBlur(srcgray,(3,3),0)
        # # edged = copy.deepcopy(blurred)
        # # edged = cv2.Canny(blurred,50,200,edged,3,False)
        # high,width,CH = cimg.shape

        # drawtangle(output_dict['detection_boxes'][0],cimg,high,width,(0,255,0))
        # drawtangle(output_dict['detection_boxes'][1],cimg,high,width,(255,0,0))
        # drawtangle(output_dict['detection_boxes'][2],cimg,high,width,(0,0,255))

        # f = fn.replace("F5X1-UP","F5X1-UP-VF")
        # cv2.imwrite(f,cimg)

def verifymbv2opencv():

    # pbfolder = './tfrec/fast_real_frozen/'
    # pbfile = pbfolder+'faster_5x1_obj.pb'
    # pbtxtfile = pbfolder+'faster_5x1_obj.pbtxt'
    # print('try to load model')
    # opencv_net = cv2.dnn.readNetFromTensorflow(pbfile,pbtxtfile)
    # print('loaded model')

    print('try to load model')
    opencv_net = cv2.cv2.dnn.readNetFromONNX('./tfrec/fast_onnx/faster_rcnn.onnx')
    print('loaded model')

    file1 = open('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\F5X1XY5.txt','r')
    lines = file1.readlines()
    for ln in lines:
        sts = ln.strip().split(';')
        fn = sts[6]

        img = cv2.imread(fn,cv2.IMREAD_COLOR)
        rows,cols,CH = img.shape
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

        print(networkOutput)

        # scorelist = []
        # print('show score..................................................')
        idx = 0
        for detection in networkOutput[0,0]:
          # print(detection)
          # idx = idx + 1
          # if idx == 3:
          #   break
        #   score = float(detection[2])
        #   scorelist.append(score)

        #   #print(scorelist)

        # #   #if score > 0.015:
          # top = detection[3]*rows
          # left = detection[4]*cols
          # bottom = detection[5]*rows
          # right = detection[6]*cols
          left = detection[3] * cols
          top = detection[4] * rows
          right = detection[5] * cols
          bottom = detection[6] * rows

        #   #draw a red rectangle around detected objects
          #if (right-left) > 200 and (bottom -top) < 80:
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
    cv2.rectangle(cimg,(pxmin,pymin),(pxmax,pymax),sc,3)


def convertmodeltopb():
  model = tf.saved_model.load('./tfrec/fast_model_frozen/saved_model')
  #model = tf.keras.models.load_model(oldmodelfn, custom_objects={'KerasLayer': hub.KerasLayer})
  #path of the directory where you want to save your model
  frozen_out_path = './tfrec/fast_real_frozen'
  # name of the .pb file
  frozen_graph_filename = 'faster_5x1_obj'
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

  tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=frozen_out_path,
                    name=f"{frozen_graph_filename}.pb",
                    as_text=False)

  tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pbtxt",
                  as_text=True)


def convertmodeltopb4():
  loaded = tf.saved_model.load('./tfrec/fast_model_frozen/saved_model')
  infer = loaded.signatures['serving_default']
  f = tf.function(infer).get_concrete_function(input_tensor=tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.uint8))
  f2 = convert_variables_to_constants_v2(f)
  graph_def = f2.graph.as_graph_def()
  # Export frozen graph
  with tf.io.gfile.GFile('./tfrec/fast_real_frozen2/faster_5x1_obj.pb', 'wb') as f:
     f.write(graph_def.SerializeToString())

def convertmodeltopb2():
  from tensorflow.python.tools import freeze_graph
  output_node_names = ['StatefulPartitionedCall']
  output_node_names = ','.join(output_node_names)
  save_pb_model_path = './tfrec/fast_real_frozen3/faster_5x1_obj.pb'
  input_saved_model_dir='./tfrec/fast_model_frozen/saved_model' 

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


verifymbv2pb9()

#convertmodeltopb2()

#verifymbv2pb4()

#verifymbv2pb()

#convertmodeltopb()

#verifymbv2opencv()

#convertmodeltopb2()

#convertmodeltopb()

#verifymbv2()

#writesiglefile('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\F5X1XY.txt','./tfrec/train_dataset.tfrecord')
#writesiglefile('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\F5X1XY2.txt','./tfrec/eval_dataset2.tfrecord')

# writesiglefile('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\SIXXY.txt','./tfrec/train_dataset_six.tfrecord')
# writesiglefile('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\SIXXY1.txt','./tfrec/eval_dataset_six.tfrecord')

# writesiglefile('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\F2X1XXY.txt','./tfrec/train_dataset_f2x1.tfrecord')
# writesiglefile('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\F2X1XXY1.txt','./tfrec/eval_dataset_f2x1.tfrecord')

# writesiglefile('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\A10XY.txt','./tfrec/train_dataset_a10.tfrecord')
# writesiglefile('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\A10XY1.txt','./tfrec/eval_dataset_a10.tfrecord')

# writesiglefile('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\IIVI-XY.txt','./tfrec/train_dataset_iivi.tfrecord')
# writesiglefile('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\IIVI-XY1.txt','./tfrec/eval_dataset_iivi.tfrecord')

#TF2.4

#TRAIN COMMAND

#python.exe object_detection/model_main_tf2.py  --pipeline_config_path=D:\PlanningForCast\condaenv\tfrec\faster_rcnn101\pipeline.config 
# --model_dir=D:\PlanningForCast\condaenv\tfrec\faster_model --num_train_steps=5000 --checkpoint_every_n=100  --alsologtostderr

# python object_detection/exporter_main_v2.py --input_type=image_tensor  --pipeline_config_path=./pipeline.config  
# --trained_checkpoint_dir=./faster_model  --output_directory=./fast_model_frozen


# python.exe object_detection/model_main_tf2.py 
# --pipeline_config_path=D:\PlanningForCast\condaenv\tf_models_xxx\research\object_detection\configs\tf2\ssd_mv2_pipeline.config 
# --model_dir=D:\PlanningForCast\condaenv\tfrec\mbv3model --num_train_steps=500 --checkpoint_every_n=50  --alsologtostderr


#python object_detection/exporter_main_v2.py --input_type=image_tensor  --pipeline_config_path=D:\PlanningForCast\condaenv\tf_models_xxx\research\object_detection\configs\tf2\ssd_mv2_pipeline.config 
# --trained_checkpoint_dir=D:\PlanningForCast\condaenv\tfrec\mbv2model 
# --output_directory=D:\PlanningForCast\condaenv\tfrec\mbv2frozen 



#python.exe object_detection/model_main_tf2.py 
#--pipeline_config_path=D:\PlanningForCast\condaenv\tf_models_xxx\research\object_detection\configs\tf2\ssd_mv2_pipeline.config 
# --model_dir=./mv2/saved_model --checkpoint_dir=D:\PlanningForCast\condaenv\tfrec\mbv3model   --alsologtostderr

#python -m tf2onnx.convert --saved-model ./tfrec/fast_model_frozen/saved_model --output ./tfrec/fast_onnx/faster_rcnn.onnx --opset 11

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
#output=./tfrec/txt/frozen_graph.pbtxt --config=D:/PlanningForCast/condaenv/tfrec/ssd_mbv2_tf1/pipeline.config

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