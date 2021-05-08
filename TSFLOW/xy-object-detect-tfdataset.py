import tensorflow as tf
from object_detection.utils import dataset_util

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

writesiglefile('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\F5X1XY.txt','./tfrec/train_dataset.tfrecord')
#writesiglefile('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\F5X1XY2.txt','./tfrec/eval_dataset2.tfrecord')

#TRAIN COMMAND
# python.exe object_detection/model_main_tf2.py 
# --pipeline_config_path=D:\PlanningForCast\condaenv\tf_models_xxx\research\object_detection\configs\tf2\ssd_mv2_pipeline.config 
# --model_dir=D:\PlanningForCast\condaenv\tfrec\mbv3model --num_train_steps=500 --checkpoint_every_n=50  --alsologtostderr



#python.exe object_detection/model_main_tf2.py 
#--pipeline_config_path=D:\PlanningForCast\condaenv\tf_models_xxx\research\object_detection\configs\tf2\ssd_mv2_pipeline.config 
# --model_dir=./mv2/saved_model --checkpoint_dir=D:\PlanningForCast\condaenv\tfrec\mbv3model   --alsologtostderr