import json
import sys
import cv2
sys.path.append('../')
import Rect

filename="../../personnel-records/1956/scans/parsed/pr1956_f115_4.tif"
rectfile="../../personnel-records/1956/seg/row_rect/pr1956_f0115_4_0"
outputpath="../../personnel-records/1956/visualization/pr1956_f0115_4_0.png"
#import pdb;pdb.set_trace()
img = cv2.imread(filename)

with open(rectfile) as rectjson:
    rect = json.load(rectjson)

warped,_=Rect.CropRect(img,rect)
print("writing results to "+outputpath)
cv2.imwrite(outputpath,warped)
