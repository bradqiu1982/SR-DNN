
from PIL import Image
import base64
import io
import numpy as np
import pyodbc

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

imgstr = []
labs = []

with pyodbc.connect(Driver='{ODBC Driver 17 for SQL Server}',Server='wuxinpi.china.ads.finisar.com', UID='WATApp', PWD='WATApp@123', Database='WAT') as conn:
	cursor = conn.cursor()
	sql = "select top 2 [TrainingImg],[ImgVal] from [WAT].[dbo].[AITrainingData] where Revision = 'OGP-IIVI'"
	cursor.execute(sql)
	rows = cursor.fetchall() 
	for row in rows:
		imgstr.append(str(row[0]))
		labs.append(str(row[1]))
	cursor.close()

bt = base64.b64decode(imgstr[0])
im = Image.open(io.BytesIO(bt))
img1 = np.array(im)

bt = base64.b64decode(imgstr[1])
im = Image.open(io.BytesIO(bt))
img2 = np.array(im)

imgarray = np.vstack((img1[None],img2[None]))

imgarray = np.vstack((imgarray,img2[None]))

print(imgarray.shape)

labarray = np.array(labs)
print(labarray.shape)

