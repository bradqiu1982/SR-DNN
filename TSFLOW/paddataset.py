
import pathlib
import random
import matplotlib.pyplot as plt
import os
from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from object_detection.utils import dataset_util

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import numpy as np

import cv2
import copy
import uuid



def convertimag2():
	data_root = pathlib.Path('D:/PlanningForCast/condaenv/PAD2/Vespa')
	all_image_paths = list(data_root.glob('*/*/*'))

	all_image_paths = [str(path) for path in all_image_paths]
	for fn in all_image_paths:
		if ('.png' in fn) and ('2021' in fn) and (('after' in fn) or ('before' in fn)):
			img = cv2.imread(fn,cv2.IMREAD_COLOR)
			img = cv2.resize(img,(1024,1024))
			
			dt = str(pathlib.Path(fn).parents[0]).replace(str(pathlib.Path(fn).parents[1]),'').replace(' ','_').replace('\\','').replace('-','')
			nm = pathlib.Path(fn).name
			nfile = './PAD2/VespaCovt/'+dt+'_'+nm.replace('.png','.jpg')

			cv2.imwrite(nfile,img,[cv2.IMWRITE_JPEG_QUALITY,100])


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


def drawtangle(box,cimg,high,width):
	ymin = box[0]
	xmin = box[1]
	ymax = box[2]
	xmax = box[3]
	pxmin = int(xmin*width)
	pymin = int(ymin*high)
	pxmax = int(xmax*width)
	pymax = int(ymax*high)

	fxmin = 90
	fxmax = 1024-90
	fymin = 90
	fymax = 1024-90
	xmid = (pxmin+pxmax)/2
	ymid = (pymin+pymax)/2

	if (xmid < fxmin or xmid > fxmax or ymid < fymin or ymid > fymax):
		cv2.rectangle(cimg,(pxmin,pymin),(pxmax,pymax),(0,0,255),3)
	else:
		cv2.rectangle(cimg,(pxmin,pymin),(pxmax,pymax),(0,255,0),3)

def verifypad():
    model = tf.saved_model.load('./tfrec/orion_savedmodel/saved_model')

    data_root = pathlib.Path('D:/PlanningForCast/condaenv/PAD2/shortcheck1')
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    for fn in all_image_paths:
        if '.jpg' in fn:

            # img = tf.io.read_file(fn)
            # image_tensor = tf.io.decode_image(img, channels=3)
            # image_tensor  = tf.image.resize(image_tensor ,[1024,1024])
            # image_tensor = tf.expand_dims(image_tensor, axis=0)
            # output_dict = model(image_tensor)
            # print(output_dict['detection_boxes'], file=open("./res_bx.txt", "a"))
            # print(output_dict['detection_classes'], file=open("./res_cla.txt", "a"))
            #print(output_dict['detection_scores'], file=open("./res_score.txt", "a"))
            # print(output_dict['detection_multiclass_scores'], file=open("./res_mulscore.txt", "a"))

            cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
            cimg = cv2.resize(cimg,(1024,1024))
            high,width,CH = cimg.shape

            img_tensor = copy.deepcopy(cimg)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            output_dict = model(img_tensor)

            for i in range(100):
                if float(output_dict['detection_scores'][0][i]) > 0.05:
                    drawtangle(output_dict['detection_boxes'][0][i],cimg,high,width)
            cv2.imwrite(fn.replace('.jpg','_detect.jpg'),cimg)



def getdetectboxes(fn,model):
	boxes = []
	cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
	cimg = cv2.resize(cimg,(1024,1024))
	#high,width,CH = cimg.shape

	img_tensor = copy.deepcopy(cimg)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	output_dict = model(img_tensor)

	for i in range(100):
		if float(output_dict['detection_scores'][0][i]) > 0.05:
			boxes.append(output_dict['detection_boxes'][0][i])
	return boxes

def getfilterboxes(boxes):
	x = 0
	filteredbox = []
	for box in boxes:
		ymin = box[0]
		xmin = box[1]
		ymax = box[2]
		xmax = box[3]
		pxmin = int(xmin*1024)
		pymin = int(ymin*1024)
		pxmax = int(xmax*1024)
		pymax = int(ymax*1024)

		fxmin = 90
		fxmax = 1024-90
		fymin = 90
		fymax = 1024-90

		xmid = (pxmin+pxmax)/2
		ymid = (pymin+pymax)/2

		if (xmid < fxmin or xmid > fxmax or ymid < fymin or ymid > fymax):
			x = 1
		else:
			filteredbox.append(box)
	return filteredbox

def compareboxes_(beforeboxes,afterboxes):
	newbox = []
	for abox in afterboxes:
		matched = 0
		for bbox in beforeboxes:
			aymin = abox[0]
			axmin = abox[1]
			aymax = abox[2]
			axmax = abox[3]

			apxmin = int(axmin*1024)
			apymin = int(aymin*1024)
			apxmax = int(axmax*1024)
			apymax = int(aymax*1024)

			axmid = (apxmin+apxmax)/2
			aymid = (apymin+apymax)/2

			bymin = bbox[0]
			bxmin = bbox[1]
			bymax = bbox[2]
			bxmax = bbox[3]

			bpxmin = int(bxmin*1024)-10
			bpymin = int(bymin*1024)-10
			bpxmax = int(bxmax*1024)+10
			bpymax = int(bymax*1024)+10

			if (axmid >= bpxmin and axmid <= bpxmax and aymid >= bpymin and aymid <= bpymax):
				matched = 1
				break
		if(matched == 0):
			newbox.append(abox)

	return newbox


def compareboxes(beforeboxes,afterboxes):
	beforelen = len(beforeboxes)
	afterlen = len(afterboxes)

	beforefilterbx = getfilterboxes(beforeboxes)
	afterfilterbx = getfilterboxes(afterboxes)

	beforefltlen = len(beforefilterbx)
	afterfltlen = len(afterfilterbx)

	print('len-'+' al:'+str(afterlen)+' bl:'+str(beforelen)+' afl:'+str(afterfltlen)+' bfl:'+str(beforefltlen))

	newbox = []
	if (afterlen - beforelen) == 1:
		if(afterfltlen -beforefltlen) == 1:
			newbox = compareboxes_(beforefilterbx,afterfilterbx)
		else:
			newbox = compareboxes_(beforeboxes,afterboxes)
	else:
		if(afterfltlen -beforefltlen) == 1:
			newbox = compareboxes_(beforefilterbx,afterfilterbx)

	return newbox

	# if(len(newbox) == 1):
	# 	return newbox

	# return []

def drawnewmark(bimg,aimg,newcheckboxes):
	for box in newcheckboxes:
		ymin = box[0]
		xmin = box[1]
		ymax = box[2]
		xmax = box[3]
		pxmin = int(xmin*1024)
		pymin = int(ymin*1024)
		pxmax = int(xmax*1024)
		pymax = int(ymax*1024)
		cv2.rectangle(bimg,(pxmin,pymin),(pxmax,pymax),(0,255,0),3)
		cv2.rectangle(aimg,(pxmin,pymin),(pxmax,pymax),(0,255,0),3)

def scannormalpad(srcthr):

	high,width = srcthr.shape
	lowy = 10
	highy = high-10
	
	lowx = 5
	highx = width-5

	xlist = []
	ylist = []

	for x in range(5,width-5):
		submat = srcthr[lowy:highy,x:x+1]
		nonzero = cv2.countNonZero(submat)
		if nonzero >= 2:
			xlist.append(x)
	
	if(len(xlist) == 0):
		for x in range(5,width-5):
			submat = srcthr[lowy:highy,x:x+1]
			nonzero = cv2.countNonZero(submat)
			if nonzero >= 1:
				xlist.append(x)

	for y in range(10,high-10):
		submat = srcthr[y:y+1,lowx:highx]
		nonzero = cv2.countNonZero(submat)
		if nonzero >= 2:
			ylist.append(y)

	if(len(ylist) == 0):
		for y in range(10,high-10):
			submat = srcthr[y:y+1,lowx:highx]
			nonzero = cv2.countNonZero(submat)
			if nonzero >= 1:
				ylist.append(y)

	return xlist,ylist

def scangoldpad(padimg,fn):

	srcgray = cv2.cvtColor(padimg,cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(srcgray,(3,3),0)
	srcthr = copy.deepcopy(blurred)
	srcthr = cv2.Canny(blurred,50,200,srcthr,3,False)

	high,width = srcthr.shape
	lowy = 1
	highy = high-1
	
	lowx = 1
	highx = width-1

	xlist = []
	ylist = []

	for x in range(1,width-2):
		submat = srcthr[lowy:highy,x:x+1]
		nonzero = cv2.countNonZero(submat)
		if nonzero >= 2:
			xlist.append(x)
	
	for y in range(1,high-2):
		submat = srcthr[y:y+1,lowx:highx]
		nonzero = cv2.countNonZero(submat)
		if nonzero >= 2:
			ylist.append(y)

	lowy = np.min(ylist)+5
	highy = np.max(ylist)-5

	lowx = np.min(xlist)+5
	highx = np.max(xlist)-5

	padimg = cv2.detailEnhance(padimg)
	srcgray = cv2.cvtColor(padimg,cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(srcgray,(3,3),0)
	srcthr = copy.deepcopy(blurred)
	srcthr = cv2.Canny(blurred,50,200,srcthr,3,False)

	# cv2.imwrite(fn.replace('.jpg','_edged.jpg'),srcthr)

	xlist = []
	ylist = []

	start = 0
	for y in range(lowy,highy):
		submat = srcthr[y:y+1,lowx:highx]
		nonzero = cv2.countNonZero(submat)

		if nonzero == 0 and start == 0:
			start = 1
		if nonzero == 0 and start == 1 and len(ylist) > 0:
			break
		if nonzero >= 1 and start == 1:
			ylist.append(y)

	if len(ylist) > 0:
		start = 0
		highy = np.max(ylist)
		for x in range(lowx,highx):
			submat = srcthr[lowy:highy,x:x+1]
			nonzero = cv2.countNonZero(submat)
			if nonzero == 0 and start == 0:
				start = 1
			if nonzero == 0 and start == 1 and len(xlist) > 0:
				break
			if nonzero >= 1 and start == 1:
				xlist.append(x)

	return xlist,ylist


def findmarkpointbox(aimg,box,fn):
	retbox = []
	ymin = box[0]
	xmin = box[1]
	ymax = box[2]
	xmax = box[3]
	pxmin = int(xmin*1024)
	pymin = int(ymin*1024)
	pxmax = int(xmax*1024)
	pymax = int(ymax*1024)

	# print('pad width:'+str(pxmax-pxmin)+' pad high:'+str(pymax-pymin))

	padimg = aimg[pymin:pymax,pxmin:pxmax]

	# sharpimg = cv2.GaussianBlur(padimg,(0,0),3)
	# sharpimg = cv2.addWeighted(padimg,2.0,sharpimg,-0.4,0)

	srcgray = cv2.cvtColor(padimg,cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(srcgray,(3,3),0)
	
	retval,srcthr = cv2.threshold(blurred,220,255,cv2.THRESH_BINARY)

	xlist,ylist = scannormalpad(srcthr)

	if len(xlist) > 0 and len(ylist) > 0:
		markwidth = np.max(xlist)-np.min(xlist)
		markhigh = np.max(ylist)-np.min(ylist)
		if(markwidth >20 or markhigh > 40):
			xlist,ylist = scangoldpad(padimg,fn)
			if len(xlist) > 0 and len(ylist) > 0:
				cv2.rectangle(aimg,(pxmin+np.min(xlist)-2,pymin+np.min(ylist)-2),(pxmin+np.max(xlist)+2,pymin+np.max(ylist)+2),(0,0,255),2)
		else:
			cv2.rectangle(aimg,(pxmin+np.min(xlist)-2,pymin+np.min(ylist)-2),(pxmin+np.max(xlist)+2,pymin+np.max(ylist)+2),(0,0,255),2)


def detectnewmark():
	model = tf.saved_model.load('./tfrec/orion_savedmodel/saved_model')

	data_root = pathlib.Path('D:/PlanningForCast/condaenv/PAD2/shortcheck1')
	all_image_paths = list(data_root.glob('*'))
	all_image_paths = [str(path) for path in all_image_paths]
	for fn in all_image_paths:
		if '.jpg' in fn and 'before' in fn:

			print(fn)

			afterfn = fn.replace('before','after')
			if pathlib.Path(afterfn).exists():
				beforeboxes = getdetectboxes(fn,model)
				afterboxes = getdetectboxes(afterfn,model)
				newmarkboxes = compareboxes(beforeboxes,afterboxes)

				bimg = cv2.imread(fn,cv2.IMREAD_COLOR)
				bimg = cv2.resize(bimg,(1024,1024))

				aimg = cv2.imread(afterfn,cv2.IMREAD_COLOR)
				aimg = cv2.resize(aimg,(1024,1024))

				if(len(newmarkboxes) == 1):
					findmarkpointbox(aimg,newmarkboxes[0],fn)

				drawnewmark(bimg,aimg,newmarkboxes)
				cv2.imwrite(fn.replace('.jpg','_detect.jpg'),bimg)
				cv2.imwrite(afterfn.replace('.jpg','_detect.jpg'),aimg)

detectnewmark()

#convertimag2()

#verifypad()

# writesiglefile('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\OrionTrain.txt','./tfrec/Orion_train_dataset_new.tfrecord')

# writesiglefile('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\OrionVerify.txt','./tfrec/Orion_eval_dataset_new.tfrecord')

# python object_detection/exporter_main_v2.py --input_type=image_tensor  --pipeline_config_path=D:\PlanningForCast\condaenv\tfrec\faster_rcnn_resnet101_1024\pipeline.config  
#  --trained_checkpoint_dir=D:\PlanningForCast\condaenv\tfrec\orion_model  --output_directory=D:\PlanningForCast\condaenv\tfrec\orion_savedmodel

