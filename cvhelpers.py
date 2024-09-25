import mediapipe as mp
import cv2
import time
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
import numpy as np

class TrackFPS:  # Weighted average (low pass) filter for frames per second
    def __init__(self, dataWeight):
        self.dw = dataWeight
        self.state = 0

    def getFPS(self):
        if self.state == 0:
            self.average = 0
            self.tlast = time.time()
            self.state = 1
        elif self.state == 1:
            self.tDelta = time.time() - self.tlast
            self.average = 1 / self.tDelta
            self.tlast = time.time()
            self.state = 2
        else:
            self.tDelta = time.time() - self.tlast
            self.fps = 1 / self.tDelta
            self.average = (self.dw * self.fps) + ((1 - self.dw) * self.average)
            self.tlast = time.time()
        return self.average


class cvGUI:  # https://tkdocs.com/index.html
    imgClick = (0, 0)

    def __init__(self, master, h, w):
        self.h = h
        self.w = w
        self.root = master
        # set up GUI
        guiFrame = ttk.Frame(self.root, padding=10)
        guiFrame.grid()
        # set control frame above video
        controlFrame = ttk.Frame(guiFrame, padding=10)
        controlFrame.grid(column=0, row=0)
        # set control

        # set video frame below controls
        imageFrame = ttk.Frame(guiFrame)
        imageFrame.grid(column=0, row=1)
        self.camIMG = ttk.Label(imageFrame)
        self.camIMG.grid(row=0, column=0)
        self.camIMG.bind('<1>', self.IMGxy)
        self.npIMG = ttk.Label(imageFrame)
        self.npIMG.grid(row=0, column=1)

    def quitGUI(self, *args):
        self.cam.release()
        self.root.destroy()

    def camStart(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        self.cam.set(cv2.CAP_PROP_FPS, 30)

    def displayImg(self, frame):
        frameRGBA = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(frameRGBA)
        imgtk = ImageTk.PhotoImage(image=img)
        self.camIMG.imgtk = imgtk
        self.camIMG.configure(image=imgtk)

    def IMGxy(self, e):  # update x,y if click in image
        cvGUI.imgClick = (e.x, e.y)
    def displayNP(self, npframe):
        frameRGBA = cv2.cvtColor(npframe, cv2.COLOR_BGR2RGBA)
        npimg = Image.fromarray(frameRGBA)
        npimgtk = ImageTk.PhotoImage(image=npimg, width=self.w, height=self.h)
        self.npIMG.imgtk = npimgtk
        self.npIMG.configure(image=npimgtk)