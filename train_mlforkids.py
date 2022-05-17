#!/usr/bin/python3

from mlforkids import MLforKidsImageProject
import requests # request img from web
import shutil # save img locally
import time
import os
import cv2
from playsound import playsound

model_filename = "ml_model"
key = "fffd0bd0-d526-11ec-8698-f9b487af9cd716d0b740-e00d-43fc-8a6f-e091123fee2e"

# this will train your model and might take a little while
myproject = MLforKidsImageProject(key)
myproject.train_model()
myproject.save_model(model_filename)


print ("\n\n\n Saved model to %s" % (model_filename))
