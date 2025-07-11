import tkinter
import PIL.Image, PIL.ImageTk
from tkinter import simpledialog
import time
from tkinter import messagebox
import os
from tkinter import *
import random
from datetime import date
from PIL import Image
import numpy as np
import cv2
import torch
import smtplib
import os
import face_recognition
from tkinter import messagebox
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
def cardDetection(self):
        global capture_img
        option = 0
        ret, frame = self.vid.get_frame()
        img = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray,1.3,5)
        print("Found {0} faces!".format(len(faces)))
        for (x, y, w, h) in faces:
            img = cv2.resize(frame, (512, 512))
            results = self.model(img)
            results.xyxy[0]  # im predictions (tensor)
            out = results.pandas().xyxy[0]  # im predictions (pandas)
            if len(out) > 0:
                xmin = int(out['xmin'].ravel()[0])
                ymin = int(out['ymin'].ravel()[0])
                xmax = int(out['xmax'].ravel()[0])
                ymax = int(out['ymax'].ravel()[0])
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                self.tf1.delete(0,END)
                self.tf1.insert(0,"Card Detected")
                cv2.imshow("Capture Face",img)
                cv2.waitKey(0)
            else:
                small_frame = cv2.resize(frame, (800, 800))
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB color space
                face_locations = face_recognition.face_locations(rgb_small_frame)  # Locate faces in the frame
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  # Encode faces in the frame
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.encodings, face_encoding)  # Compare face encodings
                    face_distance = face_recognition.face_distance(self.encodings, face_encoding)  # Calculate face distance
                    best_match_index = np.argmin(face_distance)  # Get the index of the best match
                    if matches[best_match_index]:  # If the face is a match
                        name = self.names[best_match_index]  # Get the corresponding name
                        if name in self.names:
                            self.tf1.delete(0,END)
                            self.tf1.insert(0,"Card Not Detected")
                            self.tf2.delete(0,END)
                            self.tf2.insert(0,name)
                        else:
                            self.tf1.delete(0,END)
                            self.tf1.insert(0,"Card Not Detected")
                            self.tf2.delete(0,END)
                            self.tf2.insert(0,"Unable to Recognize")
    