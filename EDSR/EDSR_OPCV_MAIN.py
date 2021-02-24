import cv2
from cv2 import dnn_superres
 sr = dnn_superres.DnnSuperResImpl_create()
 sr.readModel("./test/EDSR_x4.pb")
 sr.setModel("edsr",4)
 img = cv2.imread("./test/bd.png")
 res = sr.upsample(img)
 cv2.imwrite("./test/bd4.png",res)
 