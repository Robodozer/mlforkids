#!/usr/bin/python3

from mlforkids import MLforKidsImageProject
import requests # request img from web
import shutil # save img locally
import time
import os
from playsound import playsound

url = input('Please enter an image URL (string):') #prompt user for img url
file_name = "/home/guest/mlforkids/MLforkids%s.jpg" % (time.strftime("%M%S"))
class bcolors:
    OK = '\033[92m' #GREEN
    WARNING = '\033[93m' #YELLOW
    FAIL = '\033[91m' #RED
    RESET = '\033[0m' #RESET COLOR
res = requests.get(url, stream = True)

if res.status_code == 200:
    with open(file_name,'wb') as f:
        shutil.copyfileobj(res.raw, f)
    print(bcolors.OK + 'Image sucessfully Downloaded: ' + bcolors.RESET,file_name)
else:
    print(bcolors.FAIL +'Image Couldn\'t be retrieved'+bcolors.RESET)
# treat this key like a password and keep it secret!
key = "fffd0bd0-d526-11ec-8698-f9b487af9cd716d0b740-e00d-43fc-8a6f-e091123fee2e"

# this will train your model and might take a little while
myproject = MLforKidsImageProject(key)
myproject.train_model()

# CHANGE THIS to the image file you want to recognize
demo = myproject.prediction(file_name)

label = demo["class_name"]
confidence = demo["confidence"]
os.remove(file_name)
print ("Deleted File" + bcolors.RESET)
playsound('BEEP.mp3')
playsound('TCS.wav')
# CHANGE THIS to do something different with the result
print (bcolors.OK + "\n |------------------------------------------------|")
print (" |      RESULT: '%s' with %d%% confidence     |" % (label, confidence))
print (" |------------------------------------------------|")