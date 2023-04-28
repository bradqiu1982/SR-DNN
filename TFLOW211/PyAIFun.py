
import os
import pathlib
import numpy as np
import cv2
import copy
import sys
import math


import tensorflow as tf
from object_detection.utils import dataset_util

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config

from tensorflow.python.client import session

import json
from flask import *

from multiprocessing import Lock

mylock = Lock()
cache = dict()
# initialize the flask application
app = Flask(__name__)


def getXRayOBJDetectModel():
	if 'XRAY' not in cache:
		model = tf.saved_model.load('./ObjectDetectModel/xray_savemodel_215_good/saved_model')
		cache['XRAY'] = model
		return model
	else:
		return cache['XRAY']


@app.route("/XRAYOBJDetect", methods=["POST"])
def XRAYOBJDetect():
	try:
		res = []
		request_json = request.get_json()
		ipth = request_json['imgpath']

		despath = ipth.replace('\\SRC\\','\\DES\\')
		if not pathlib.Path(despath).is_dir():
			os.mkdir(despath)

		model =  getXRayOBJDetectModel()
		data_root = pathlib.Path(ipth)
		all_image_paths = list(data_root.glob('*'))
		all_image_paths = [str(path) for path in all_image_paths]
		for fn in all_image_paths:
			if '.JPG' in fn.upper() or '.JPEG' in fn.upper():

				mylock.acquire()
				try:
					print(fn)
					cimg = cv2.imread(fn,cv2.IMREAD_COLOR)
					cimg = cv2.resize(cimg,(1280,920))
					high,width,CH = cimg.shape
					img_tensor = copy.deepcopy(cimg)
					img_tensor = np.expand_dims(img_tensor, axis=0)
					output_dict = model(img_tensor)

					warninginfos = solveXRAYImg(fn,output_dict,cimg)
					for item in warninginfos:
						res.append(item)
				except:
					exception_message = sys.exc_info()[1]
					print(str(exception_message))
				finally:
					mylock.release()

		response = jsonify(res)
		response.status_code = 200
	except:
		exception_message = sys.exc_info()[1]
		print(str(exception_message))
		response = jsonify({"content":str(exception_message)})
		response.status_code = 400
	return response

def ImageType(gydelta,gyavg):
	if gydelta > 100:
		return 1
	else:
		if gyavg < 90:
			return 2
		else:
			return 3

def getRealBond(cimg,box):
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

	if oleft < 0:
		oleft = 0
	if otop < 0:
		otop = 0
	if oright > width:
		oright = width-1
	if obotm > high:
		obotm = high-1

	#print( ' oleft ' +str(oleft)+' oright ' +str(oright)+'  otop ' +str(otop)+'  obotm  ' +str(obotm))

	subcimg = cimg[otop:obotm,oleft:oright]
	srcgray = cv2.cvtColor(subcimg,cv2.COLOR_BGR2GRAY)
	hg,wd = srcgray.shape

	gymaxval = int(np.max(srcgray.flatten()))
	gyminval = int(np.min(srcgray.flatten()))
	gyavg = int(np.average(srcgray.flatten()))
	gymidval = int((gymaxval+gyminval)/2)
	gydelta = gymaxval-gyminval

	#print('i: '+str(i)+'  maxval:  '+str(gymaxval)+' minval: '+str(gyminval) + ' delta: '+str(gydelta) + ' midval: '+str(gymidval)+' avgval: '+str(gyavg))
	#cv2.imwrite('D:/PlanningForCast/condaenv/COGA-PAD/PAD/'+'PAD_'+str(i)+'.jpg',subcimg)

	imgtype = ImageType(gydelta,gyavg)
	thrd = 100
	if imgtype == 1 or imgtype == 3:
		thrd = gymidval
		if thrd > gyavg:
			thrd = gyavg
	else:
		if imgtype == 2:
			#thrd = 80
			thrd = gymidval
			if thrd > gyavg:
				thrd = gyavg
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

	if oleft < 0:
		oleft = 0
	if otop < 0:
		otop = 0
	high,width,CH = cimg.shape
	if oright > width:
		oright = width-1
	if obotm > high:
		obotm = high-1

	return oleft,oright,otop,obotm,gydelta,gyavg


def getgasarea(cimg,oleft,oright,otop,obotm,rd,lccx,lccy,imgtype):
	if oleft < 0:
		oleft = 0
	if otop < 0:
		otop = 0
	high,width,CH = cimg.shape
	if oright > width:
		oright = width-1
	if obotm > high:
		obotm = high-1

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

	maxval = int(max(grayvals))
	minval = int(min(grayvals))
	deltaval = int(maxval-minval)
	midval = int((maxval+minval)/2)
	avg = int(sum(grayvals)/len(grayvals))

	gasval = midval
	if gasval > avg:
		gasval = avg

	# print('  max: '+str(maxval)+'  min: '+str(minval)+' midval: '+str(int((maxval+minval)/2))+' avgval: '+str(avg)+ ' delta: '+str(maxval-minval))

	if imgtype == 1 or imgtype == 3:
		gasval += int(deltaval*0.1)
		# gasval = 82
		# if gasval < 82:
		# 	gasval = 82
	elif imgtype == 2:
		gasval += int(deltaval*0.2)
		# if gasval < 78:
		# 	gasval = 78
	else:
		gasval += int(deltaval*0.1)

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


def calculateRate(desfn,cimg,box,boxcount):

	res = []

	oleft,oright,otop,obotm,gydelta,gyavg = getRealBond(cimg,box)
	imgtype = ImageType(gydelta,gyavg)

	cx = int((oleft+oright)/2)
	cy = int((otop+obotm)/2)
	lccx = int((oright-oleft)/2)
	lccy = int((obotm - otop)/2)
	hg = obotm - otop
	wd = oright - oleft

	if boxcount > 56 and hg < 30:
		return res

	# rd = wd/2
	# if hg > wd:
	# 	rd = hg/2
	# if imgtype == 2 or imgtype == 3:
	# 	rd = wd/2
	# 	if hg < wd:
	# 		rd = hg/2

	rd = wd/2
	if hg < wd:
		rd = hg/2

	#with open("D:/RADIUS.txt","a") as file:
	#	file.write(str(rd)+"\n")

	if rd < 8:
		return res

	# print('cx: '+ str(cx) + '  cy: '+str(cy)+'  rd: '+str(rd))
	
	retval = {}
	gasarea = getgasarea(cimg,oleft,oright,otop,obotm,rd,lccx,lccy,imgtype)

	solderarea = int(np.pi*rd*rd)
	rate = gasarea/solderarea*100
	ccolor = (0,255,0)

	rddict = {}

	if rd < 14.0:
		ccolor = (0,0,255)
		retval['level'] = 'WARNING'
		retval['rate'] = 'radius abnormal'
		retval['image'] = desfn
		rddict['rdfail'] = 'true'
	elif rate >= 35.0:
		ccolor = (0,0,255)
		retval['level'] = 'FAIL'
		retval['rate'] = str(rate)[0:4]
		retval['image'] = desfn
	elif rate > 25.0 and rate < 35.0:
		ccolor = (0,255,255)
		retval['level'] = 'WARNING'
		retval['rate'] = str(rate)[0:4]
		retval['image'] = desfn

	if len(retval) > 0:
		res.append(retval)

	#print(str(i)+':  solderarea: '+str(solderarea)+'   gasarea: '+str(gasarea) + '  rate: '+str(gasarea/solderarea))

	cv2.circle(cimg,(cx,cy),int(rd),ccolor,1)
	if len(rddict) > 0:
		cv2.putText(cimg,'rd:'+ str(rd),(cx+5,cy-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
	else:
		cv2.putText(cimg,str(rate)[0:4],(cx+5,cy-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

	return res


def solveXRAYImg(fn,output_dict,cimg):
	desfn = fn.replace('.jpg','_detect.jpg').replace('.jpeg','_detect.jpg').replace('\\SRC\\','\\DES\\')
	res = []
	boxcount = 0
	for i in range(100):
		if float(output_dict['detection_scores'][0][i]) >= 0.05:
			boxcount = boxcount + 1

	for i in range(100):
		if float(output_dict['detection_scores'][0][i]) >= 0.05:
			box = output_dict['detection_boxes'][0][i]
			warninginfos = calculateRate(desfn,cimg,box,boxcount)
			for item in warninginfos:
					res.append(item)

	cv2.imwrite(desfn,cimg)
	return res










def  getVCSELOBJDectModel(imgtype):
	if imgtype not in cache:
		model = tf.saved_model.load('./ObjectDetectModel/'+imgtype+'/saved_model')
		cache[imgtype] = model
		return model
	else:
		return cache[imgtype]

@app.route("/SingleOBJDetect", methods=["POST"])
def SingleOBJDetect():
	try:
		res = []
		request_json = request.get_json()
		imgtype = request_json['imgtype']

		model =  getVCSELOBJDectModel(imgtype)

		f = request_json['imgpath']
		if '.JPG' in f.upper() or '.JPEG' in f.upper():

			img = tf.io.read_file(f)
			image_tensor = tf.io.decode_image(img, channels=3)
			image_tensor = tf.expand_dims(image_tensor, axis=0)
			output_dict = model(image_tensor)
			boxes = output_dict['detection_boxes'].numpy()
			score = output_dict['detection_scores'].numpy()

			item = {}
			box = boxes[0][0]
			item['left'] = str(box[1])
			item['top'] = str(box[0])
			item['right'] = str(box[3])
			item['botm'] = str(box[2])
			item['imgname']=f
			item['score']=str(score[0][0])
			res.append(item)

			if 'IIVI' in imgtype:
				item = {}
				box = boxes[0][1]
				item['left'] = str(box[1])
				item['top'] = str(box[0])
				item['right'] = str(box[3])
				item['botm'] = str(box[2])
				item['imgname']=f
				item['score']=str(score[0][1])
				res.append(item)
				
		response = jsonify(res)
		response.status_code = 200
	except:
		exception_message = sys.exc_info()[1]
		print(str(exception_message))
		response = jsonify({"content":str(exception_message)})
		response.status_code = 400
	return response

# endpoint OBJDetect() with post method
@app.route("/OBJDetect", methods=["POST"])
def OBJDetect():
	try:
		res = []
		request_json = request.get_json()
		#print(request_json['imgpath'])
		#print(request_json['imgtype'])

		model =  getVCSELOBJDectModel(request_json['imgtype'])
		data_root = pathlib.Path(request_json['imgpath'])


		all_image_paths = list(data_root.glob('*'))
		fs = [str(path) for path in all_image_paths]
		for f in fs:
			if '.JPG' in f.upper() or '.JPEG' in f.upper():
				#print(f)
				item = {}
				img = tf.io.read_file(f)
				image_tensor = tf.io.decode_image(img, channels=3)
				image_tensor = tf.expand_dims(image_tensor, axis=0)
				output_dict = model(image_tensor.numpy())
				box = output_dict['detection_boxes'].numpy()
				box = box[0][0]
				item['left'] = str(box[1])
				item['top'] = str(box[0])
				item['right'] = str(box[3])
				item['botm'] = str(box[2])
				item['imgname']=f
				score = output_dict['detection_scores'].numpy()
				item['score']=str(score[0][0])
				#print(item['left'])
				#print(item['score'])
				res.append(item)
				
		response = jsonify(res)
		response.status_code = 200
	except:
		exception_message = sys.exc_info()[1]
		response = jsonify({"content":str(exception_message)})
		response.status_code = 400
	return response

# def  getCOCOBJDectModel():
# 	COCMODEL='COCMODEL'
# 	if COCMODEL not in cache:
# 		model = tf.saved_model.load('./ObjectDetectModel/COCLASER/RETINANET')
# 		cache[COCMODEL] = model
# 		return model
# 	else:
# 		return cache[COCMODEL]

# @app.route("/COCLASERWBDETECT", methods=["POST"])
# def COCLASERWBDETECT():
# 	HIGH = 640
# 	WIDTH = 640
# 	input_image_size = (HIGH, WIDTH)
# 	score = 0.4

# 	try:
# 		res = []
# 		request_json = request.get_json()
# 		#print(request_json['imgpath'])

# 		imported =  getCOCOBJDectModel()
# 		model_fn = imported.signatures['serving_default']

# 		idx = 0
# 		data_root = pathlib.Path(request_json['imgpath'])
# 		all_image_paths = list(data_root.glob('*'))
# 		fs = [str(path) for path in all_image_paths]
# 		for f in fs:
# 			if '.JPG' in f.upper() or '.JPEG' in f.upper():
# 				print(f)

# 				mylock.acquire()

# 				try:
# 					img = tf.io.read_file(f)
# 					img_tensor = tf.io.decode_image(img, channels=3)
# 					img_tensor, _ = resize_and_crop_image(img_tensor,input_image_size,padded_size=input_image_size,aug_scale_min=1.0,aug_scale_max=1.0)
# 					img_tensor = tf.expand_dims(img_tensor, axis=0)
# 					img_tensor = tf.cast(img_tensor, dtype = tf.uint8)

# 					output_dict = model_fn(img_tensor)
# 					for i in range(100):
# 						if float(output_dict['detection_scores'][0][i]) >= score:
# 							item = {}
# 							item['imgname'] = f
# 							item['score'] = str(float(output_dict['detection_scores'][0][i]))
# 							item['classid'] = str(int(output_dict['detection_classes'][0][i]))
# 							box = output_dict['detection_boxes'][0][i]
# 							item['top'] = str(float(box[0]))
# 							item['left'] = str(float(box[1]))
# 							item['botm'] = str(float(box[2]))
# 							item['right'] = str(float(box[3]))
# 							res.append(item)
# 				except:
# 					exception_message = sys.exc_info()[1]
# 					print(str(exception_message))
# 				finally:
# 					mylock.release()

# 				if idx > 9:
# 					break
# 				idx = idx + 1

# 		response = jsonify(res)
# 		response.status_code = 200
# 	except:
# 		exception_message = sys.exc_info()[1]
# 		response = jsonify({"content":str(exception_message)})
# 		response.status_code = 400

# 	return response


if __name__ == "__main__":
	from waitress import serve
	serve(app, host="0.0.0.0", port=5000)
#    run flask application in debug mode
	# app.run(debug=True)
