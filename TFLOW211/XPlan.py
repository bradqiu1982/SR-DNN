import pathlib
import os
from PIL import Image
import numpy as np
import cv2
import copy
import uuid

import tensorflow as tf

from official.vision.data import tfrecord_lib

# from object_detection.utils import dataset_util

# from object_detection.utils import ops as utils_ops
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util

import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import math

class imagedata():
	"""docstring for ClassName"""
	def __init__(self):
		self.height = 0
		self.width = 0
		self.filename = ''
		self.xmin = []
		self.xmax = []
		self.ymin = []
		self.ymax = []
		self.clsname = []
		self.clslabel = []


def convertimag2():
	data_root = pathlib.Path('D:/PlanningForCast/condaenv/AIdata/pad')
	all_image_paths = list(data_root.glob('*'))

	all_image_paths = [str(path) for path in all_image_paths]
	for fn in all_image_paths:
		if ('.jpeg' in fn):
			img = cv2.imread(fn,cv2.IMREAD_COLOR)
			img = cv2.resize(img,(1280,920))
			nfile = fn.replace('.jpeg','.jpg')
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


def create_tf_example1(imgdata):

    filename = b'' # Filename of the image. Empty if image is not from file

    encoded_image_data = None
    with tf.io.gfile.GFile(imgdata.filename, 'rb') as fid:
        encoded_image_data = fid.read()

    image_format = b'jpeg' # b'jpeg' or b'png'

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tfrecord_lib.convert_to_feature(imgdata.height),
        'image/width': tfrecord_lib.convert_to_feature(imgdata.width),
        'image/filename': tfrecord_lib.convert_to_feature(filename),
        'image/source_id': tfrecord_lib.convert_to_feature(filename),
        'image/encoded': tfrecord_lib.convert_to_feature(encoded_image_data),
        'image/format': tfrecord_lib.convert_to_feature(image_format),
        'image/object/bbox/xmin': tfrecord_lib.convert_to_feature(imgdata.xmin),
        'image/object/bbox/xmax': tfrecord_lib.convert_to_feature(imgdata.xmax),
        'image/object/bbox/ymin': tfrecord_lib.convert_to_feature(imgdata.ymin),
        'image/object/bbox/ymax': tfrecord_lib.convert_to_feature(imgdata.ymax),
        'image/object/class/text': tfrecord_lib.convert_to_feature(imgdata.clsname),
        'image/object/class/label': tfrecord_lib.convert_to_feature(imgdata.clslabel),
    }))
    
    return tf_example


def writesiglefile1(fn,outfile):
	imgdatas = {}
	writer = tf.io.TFRecordWriter(outfile)
	file1 = open(fn,'r')
	lines = file1.readlines()
	for ln in lines:
		sts = ln.strip().split(';')
		# ih,iw,xmin,xmax,ymin,ymax,fpath,clatx,claid
		# tf_example = create_tf_example(int(sts[0]),int(sts[1]),float(sts[2]),float(sts[3]),float(sts[4]),float(sts[5]),sts[6],sts[7],int(sts[8]))
		if sts[6] in imgdatas:
			imgdt = imgdatas[sts[6]]
			imgdt.xmin.append(float(sts[2])/float(sts[1]))
			imgdt.xmax.append(float(sts[3])/float(sts[1]))
			imgdt.ymin.append(float(sts[4])/float(sts[0]))
			imgdt.ymax.append(float(sts[5])/float(sts[0]))
			imgdt.clsname.append(bytes(sts[7],'utf-8'))
			imgdt.clslabel.append(int(sts[8]))
		else:
			imgdt = imagedata()
			imgdt.height = int(sts[0])
			imgdt.width = int(sts[1])
			imgdt.filename = sts[6]
			imgdt.xmin.append(float(sts[2])/float(sts[1]))
			imgdt.xmax.append(float(sts[3])/float(sts[1]))
			imgdt.ymin.append(float(sts[4])/float(sts[0]))
			imgdt.ymax.append(float(sts[5])/float(sts[0]))
			imgdt.clsname.append(bytes(sts[7],'utf-8'))
			imgdt.clslabel.append(int(sts[8]))
			imgdatas[sts[6]] = imgdt
	file1.close()

	for k,v in imgdatas.items():
		tf_example = create_tf_example1(v)
		writer.write(tf_example.SerializeToString())
	writer.close()


def drawtangle(box,cimg,high,width,i):
	ymin = box[0]
	xmin = box[1]
	ymax = box[2]
	xmax = box[3]
	pxmin = int(xmin*width)
	pymin = int(ymin*high)
	pxmax = int(xmax*width)
	pymax = int(ymax*high)
	
	#bimg = cimg[pymin-4:pymax+4,pxmin-4:pxmax+4]
	#cv2.imwrite('D:/PlanningForCast/condaenv/COGA-PAD/PAD/'+'PAD_'+str(i)+'.jpg',bimg)

	cv2.rectangle(cimg,(pxmin,pymin),(pxmax,pymax),(0,255,0),3)

def ImageType(gydelta,gyavg):
	if gydelta > 100:
		return 1
	else:
		if gyavg < 90:
			return 2
		else:
			return 3

def getRealBond(cimg,box,i):
	high,width,CH = cimg.shape
	ymin = box[0]
	xmin = box[1]
	ymax = box[2]
	xmax = box[3]
	pxmin = int(xmin*width)
	pymin = int(ymin*high)
	pxmax = int(xmax*width)
	pymax = int(ymax*high)
	
	oleft = pxmin-4
	oright = pxmax+4
	otop = pymin-4
	obotm = pymax+4


	subcimg = cimg[otop:obotm,oleft:oright]
	srcgray = cv2.cvtColor(subcimg,cv2.COLOR_BGR2GRAY)
	hg,wd = srcgray.shape

	gymaxval = int(np.max(srcgray.flatten()))
	gyminval = int(np.min(srcgray.flatten()))
	gyavg = int(np.average(srcgray.flatten()))
	#gymidval = int((gymaxval+gyminval)/2)
	gydelta = gymaxval-gyminval

	#print('i: '+str(i)+'  maxval:  '+str(gymaxval)+' minval: '+str(gyminval) + ' delta: '+str(gydelta) + ' midval: '+str(gymidval)+' avgval: '+str(gyavg))
	#cv2.imwrite('D:/PlanningForCast/condaenv/COGA-PAD/PAD/'+'PAD_'+str(i)+'.jpg',subcimg)

	imgtype = ImageType(gydelta,gyavg)
	thrd = 100
	if imgtype == 1:
		thrd = 100
	else:
		if imgtype == 2:
			thrd = 80
		else:
			thrd = 100

	blurred = cv2.GaussianBlur(srcgray,(3,3),0)
	retval,srcthr = cv2.threshold(blurred,thrd,255,cv2.THRESH_BINARY_INV)
	
	ckleft = 0
	for x in range(0,int(wd/2),1):
		submat = srcthr[10:hg-10,x:x+1]
		nonzero = cv2.countNonZero(submat)
		if nonzero > 4:
			ckleft = x
			break
	ckleft = ckleft - 1

	ckright = 0
	for x in range(wd,int(wd/2),-1):
		submat = srcthr[10:hg-10,x-1:x]
		nonzero = cv2.countNonZero(submat)
		if nonzero > 4:
			ckright = x
			break

	ckright = ckright + 1
	ckright = wd-ckright

	cktop = 0
	for y in range(0,int(hg/2),1):
		submat = srcthr[y:y+1,10:wd-10]
		nonzero = cv2.countNonZero(submat)
		if nonzero > 4:
			cktop = y
			break
	cktop = cktop - 1

	ckbotm = 0
	for y in range(hg,int(hg/2),-1):
		submat = srcthr[y-1:y,10:wd-10]
		nonzero = cv2.countNonZero(submat)
		if nonzero > 4:
			ckbotm = y
			break

	ckbotm = ckbotm + 1
	ckbotm = hg-ckbotm

	oleft = oleft + ckleft
	oright = oright - ckright
	otop = otop + cktop
	obotm = obotm - ckbotm

	return oleft,oright,otop,obotm,gydelta,gyavg


def getgasarea(cimg,oleft,oright,otop,obotm,rd,lccx,lccy,imgtype):

	subcimg = cimg[otop:obotm,oleft:oright]
	srcgray = cv2.cvtColor(subcimg,cv2.COLOR_BGR2GRAY)
	#blurred = cv2.GaussianBlur(srcgray,(3,3),0)
	blurred = cv2.bilateralFilter(srcgray,3,75,75)
	#blurred = cv2.medianBlur(srcgray,5)
	
	grayvals = []
	yy,xx = blurred.shape
	for x in range(xx):
		for y in range(yy):
			dist = math.sqrt((x-lccx)*(x-lccx)+(y-lccy)*(y-lccy))
			if dist < float(rd)-1:
				grayvals.append(blurred[y,x])

	# np.set_printoptions(threshold=np.inf)
	# print(blurred, file=open("./srcgray.txt", "a"))
	# cv2.imwrite('./srcgray.jpg',blurred,[cv2.IMWRITE_JPEG_QUALITY,100])

	maxval = max(grayvals)
	minval = min(grayvals)

	gasval = int((maxval+minval)/2)
	avg = int(sum(grayvals)/len(grayvals))
	if avg > gasval:
		gasval = avg

	# print('min: '+str(minval)+'  max: '+str(maxval)+' gasval: '+str(gasval))

	if imgtype == 1:
		if gasval < 82:
			gasval = 82
	elif imgtype == 2:
		if gasval < 78:
			gasval = 78
	elif imgtype == 3:
		if gasval < 90:
			gasval = 90
	else:
		if gasval < 78:
			gasval = 78

	gasarea = 0
	yy,xx = blurred.shape
	for x in range(xx):
		for y in range(yy):
			val = blurred[y,x]
			if val >= gasval:
				dist = math.sqrt((x-lccx)*(x-lccx)+(y-lccy)*(y-lccy))
				if dist < float(rd)-5:
					gasarea = gasarea+1
					cimg[otop+y,oleft+x] = [255,0,0]
					#cv2.circle(cimg,(oleft+x,otop+y),0,(255,0,0),-1)
	return gasarea


def calculateRate(cimg,box,i):

	oleft,oright,otop,obotm,gydelta,gyavg = getRealBond(cimg,box,i)
	imgtype = ImageType(gydelta,gyavg)

	cx = int((oleft+oright)/2)
	cy = int((otop+obotm)/2)
	lccx = int((oright-oleft)/2)
	lccy = int((obotm - otop)/2)
	hg = obotm - otop
	wd = oright - oleft

	rd = wd/2
	if hg > wd:
		rd = hg/2
	if imgtype == 2 or imgtype == 3:
		rd = wd/2
		if hg < wd:
			rd = hg/2

	if rd < 8:
		return

	# print('cx: '+ str(cx) + '  cy: '+str(cy)+'  rd: '+str(rd))
	
	gasarea = getgasarea(cimg,oleft,oright,otop,obotm,rd,lccx,lccy,imgtype)

	solderarea = int(np.pi*rd*rd)
	rate = gasarea/solderarea*100
	ccolor = (0,255,0)
	if rate >= 35.0:
		ccolor = (0,0,255)
	elif rate > 25.0 and rate < 35.0:
		ccolor = (0,255,255)

	#print(str(i)+':  solderarea: '+str(solderarea)+'   gasarea: '+str(gasarea) + '  rate: '+str(gasarea/solderarea))

	cv2.circle(cimg,(cx,cy),int(rd),ccolor,1)
	cv2.putText(cimg,str(rate)[0:5],(cx+5,cy-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

def drawcircle(cimg,box,i):

	oleft,oright,otop,obotm,gydelta,gyavg = getRealBond(cimg,box,i)
	imgtype = ImageType(gydelta,gyavg)

	cx = int((oleft+oright)/2)
	cy = int((otop+obotm)/2)
	lccx = int((oright-oleft)/2)
	lccy = int((obotm - otop)/2)
	hg = obotm - otop
	wd = oright - oleft

	rd = wd/2
	if hg > wd:
		rd = hg/2
	if imgtype == 2 or imgtype == 3:
		rd = wd/2
		if hg < wd:
			rd = hg/2

	if rd < 8:
		return

	ccolor = (0,255,0)
	cv2.circle(cimg,(cx,cy),int(rd),ccolor,1)


def getpicturegraylevel(cimg,output_dict):
	ylist = []
	xlist = []

	high,width,CH = cimg.shape
	for i in range(100):
		if float(output_dict['detection_scores'][0][i]) >= 0.05:
			box = output_dict['detection_boxes'][0][i]
			ymin = box[0]
			xmin = box[1]
			ymax = box[2]
			xmax = box[3]
			pxmin = int(xmin*width)
			pymin = int(ymin*high)
			pxmax = int(xmax*width)
			pymax = int(ymax*high)
			xlist.append(pxmin)
			xlist.append(pxmax)
			ylist.append(pymin)
			ylist.append(pymax)

	xmin = min(xlist)
	xmax = max(xlist)
	ymin = min(ylist)
	ymax = max(ylist)

	xmid = int((xmin+xmax)/2)
	ymid = int((ymin+ymax)/2)

	srcgray = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)

	submat = srcgray[ymid-20:ymid+20,xmid+50:xmid+100]
	avg = np.average(submat.flatten())
	print('image gray level:......................'+str(int(avg)))
	return avg




def verifypad():
    model = tf.saved_model.load('./tfrec/xray_savemodel_215_good/saved_model')

    data_root = pathlib.Path('D:/PlanningForCast/condaenv/COGA-PAD/shortcheck4')
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    for fn in all_image_paths:
        if '.jpg' in fn or '.jpeg' in fn:

            # img = tf.io.read_file(fn)
            # image_tensor = tf.io.decode_image(img, channels=3)
            # image_tensor  = tf.image.resize(image_tensor ,[1024,1024])
            # image_tensor = tf.expand_dims(image_tensor, axis=0)
            # output_dict = model(image_tensor)
            # print(output_dict['detection_boxes'], file=open("./res_bx.txt", "a"))
            # print(output_dict['detection_classes'], file=open("./res_cla.txt", "a"))
            #print(output_dict['detection_scores'], file=open("./res_score.txt", "a"))
            # print(output_dict['detection_multiclass_scores'], file=open("./res_mulscore.txt", "a"))
            print(fn)
            cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
            cimg = cv2.resize(cimg,(1280,920))

            img_tensor = copy.deepcopy(cimg)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            output_dict = model(img_tensor)

            for i in range(100):
                if float(output_dict['detection_scores'][0][i]) >= 0.05:
                    calculateRate(cimg,output_dict['detection_boxes'][0][i],i)

            cv2.imwrite(fn.replace('.jpg','_detect.jpg').replace('shortcheck2','checked').replace('shortcheck3','checked').replace('shortcheck4','checked'),cimg)


def verifypad2():
    model = tf.saved_model.load('./tfrec/xray_savemodel_215_good/saved_model')

    data_root = pathlib.Path('D:/PlanningForCast/condaenv/COGA-PAD/shortcheck4')
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
            print(fn)
            cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
            cimg = cv2.resize(cimg,(1280,920))
            high,width,CH = cimg.shape

            img_tensor = copy.deepcopy(cimg)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            output_dict = model(img_tensor)

            for i in range(100):
                if float(output_dict['detection_scores'][0][i]) >= 0.04:
                	drawtangle(output_dict['detection_boxes'][0][i],cimg,high,width,i)

            cv2.imwrite(fn.replace('.jpg','_detect.jpg'),cimg)


def verifypad3():
    model = tf.saved_model.load('./tfrec/xray_savemodel_215_good/saved_model')

    data_root = pathlib.Path('D:/PlanningForCast/condaenv/COGA-PAD/shortcheck3')
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    for fn in all_image_paths:
        if '.jpg' in fn or '.jpeg' in fn:

            # img = tf.io.read_file(fn)
            # image_tensor = tf.io.decode_image(img, channels=3)
            # image_tensor  = tf.image.resize(image_tensor ,[1024,1024])
            # image_tensor = tf.expand_dims(image_tensor, axis=0)
            # output_dict = model(image_tensor)
            # print(output_dict['detection_boxes'], file=open("./res_bx.txt", "a"))
            # print(output_dict['detection_classes'], file=open("./res_cla.txt", "a"))
            #print(output_dict['detection_scores'], file=open("./res_score.txt", "a"))
            # print(output_dict['detection_multiclass_scores'], file=open("./res_mulscore.txt", "a"))
            print(fn)
            cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
            cimg = cv2.resize(cimg,(1280,920))

            img_tensor = copy.deepcopy(cimg)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            output_dict = model(img_tensor)

            for i in range(100):
                if float(output_dict['detection_scores'][0][i]) >= 0.05:
                    drawcircle(cimg,output_dict['detection_boxes'][0][i],i)

            cv2.imwrite(fn.replace('.jpg','_detect.jpg').replace('shortcheck2','checked').replace('shortcheck3','checked').replace('shortcheck4','checked'),cimg)


def test():
	a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
	yy,xx = a.shape
	for x in range(xx):
		for y in range(yy):
			print(a[y,x])


#test()

#verifypad()

#verifypad2()

#verifypad3()

#convertimag2()

writesiglefile1('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\XTRAIN.txt','./x_train_dataset.tfrecord')

#writesiglefile('\\\\wux-engsys01\\PlanningForCast\\VCSEL5\\XYFILE\\XVERIFY.txt','./tfrec/x_eval_dataset.tfrecord')


# python.exe object_detection/model_main_tf2.py  --pipeline_config_path=D:\PlanningForCast\condaenv\tfrec\faster_rcnn_resnet152_v1_800x1333\pipeline.config  
# --model_dir=D:\PlanningForCast\condaenv\tfrec\xray_model --num_train_steps=1000 --checkpoint_every_n=100  --alsologtostderr

# python object_detection/exporter_main_v2.py --input_type=image_tensor  --pipeline_config_path=D:\PlanningForCast\condaenv\tfrec\faster_rcnn_resnet152_v1_800x1333\pipeline.config  
# --trained_checkpoint_dir=D:\PlanningForCast\condaenv\tfrec\xray_model  --output_directory=D:\PlanningForCast\condaenv\tfrec\xray_savemodel
