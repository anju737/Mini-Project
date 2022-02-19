from cv2 import threshold
from Detector import *
from functools import partial
from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz"

classFile="coco.names"

threshold=0.5
detector=Detector()
detector.readClasses(classFile)


detector.downloadModel(modelURL)
detector.loadModel()

imgfile=""
vidfile=""
def browseImg():
    filename = filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes = [('Jpg Files', '*.jpg')])
    # Change label contents
    label_file_explorer1.configure(text=filename)
    imgfile=filename
def browseVid():
    filename = filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes =[("all video format", ".mp4"),("all video format", ".flv"),("all video format", ".avi")])
    # Change label contents
    label_file_explorer2.configure(text=filename)
    vidfile=filename
 # Create the root window
window = Tk()
# Set window title
window.title('Realtime Multiple Object Tracker')
# Set window size
window.geometry("1920x1280")
#Set window background color
window.config(background = "white")
button_explore1 = Button(window,text = "Browse Files",command = browseImg)
button_explore2 = Button(window,text = "Browse Files",command = browseVid)
button_image = Button(window,text = "submit",command =partial(detector.predictImage, imgfile,threshold))
button_video = Button(window,text = "submit",command = partial(detector.predictVideo, vidfile,threshold))
button_live = Button(window,text = "Live Camera",command = partial(detector.predictVideo, 0,threshold))
button_exit = Button(window,text = "Exit",command = exit)
label_file_explorer1 = Label(window, text =imgfile,width = 100, height = 4,fg = "red")
label_file_explorer2= Label(window, text =vidfile,width = 100, height = 4,fg = "red")
# Grid method is chosen for placing
# the widgets at respective positions
# in a table like structure by
# specifying rows and columns
label_file_explorer1.grid(column = 1, row = 1)
button_image.grid(column = 2, row = 3)
button_video.grid(column = 2, row = 4)
label_file_explorer2.grid(column = 1, row = 2)
button_explore1.grid(column = 1, row = 3)
button_explore2.grid(column = 1, row = 4)
button_live.grid(column = 1, row = 5)
button_exit.grid(column = 1,row = 6)
  
# Let the window wait for any events
window.mainloop()


#imagePath="Test/img4.jpeg"
#videopath=0
#threshold=0.5
#detector=Detector()
#detector.readClasses(classFile)

#detector.downloadModel(modelURL)
#detector.loadModel()

#detector.predictImage(imagePath,threshold)
#detector.predictVideo(videopath,threshold)