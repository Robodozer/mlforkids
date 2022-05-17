#!/usr/bin/python3

from Fetch import MLforKidsImageProject
import requests # request img from web
import shutil # save img locally
import time
import os
import cv2
from playsound import playsound
cap = cv2.VideoCapture(0)
ret, pic = cap.read()
cv2.imwrite("POGE.jpg", pic)
file_name = "POGE.jpg"


model_filename = "ml_model"
key = "fffd0bd0-d526-11ec-8698-f9b487af9cd716d0b740-e00d-43fc-8a6f-e091123fee2e"

myproject = MLforKidsImageProject(key)
myproject.load_model(model_filename)

# CHANGE THIS to the image file you want to recognize
demo = myproject.prediction(file_name)

label = demo["class_name"]
confidence = demo["confidence"]
# os.remove(file_name)
# print ("Deleted File")
playsound('BEEP.mp3')
playsound('TCS.wav')
# CHANGE THIS to do something different with the result
print ("\n |------------------------------------------------|")
print (" |     RESULT: '%s' with %d%% confidence     |" % (label, confidence))
print (" |------------------------------------------------|")

