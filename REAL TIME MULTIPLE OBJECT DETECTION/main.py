from cv2 import threshold
from Detector import *

modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz"

classFile="coco.names"
imagePath="Test/img4.jpeg"
videopath=0
threshold=0.5
detector=Detector()
detector.readClasses(classFile)

detector.downloadModel(modelURL)
detector.loadModel()
detector.predictImage(imagePath,threshold)
#detector.predictVideo(videopath,threshold)