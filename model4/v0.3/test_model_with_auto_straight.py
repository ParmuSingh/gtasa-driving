# test_model.py
import sys
sys.path.insert(0, "E:/workspace_py/gtasa_driving/keyloggers")
sys.path.insert(0, "E:/workspace_py/gtasa_driving/")
sys.path.insert(0, "E:/workspace_py/gtasa_driving/model4/")
import numpy as np
from sentdex_grabscreen import grab_screen
import cv2
import time
from directKeys import PressKey,ReleaseKey, W, A, S, D
from alexnet import alexnet
from sentdex_getkeys import key_check

import random

WIDTH = 160
HEIGHT = 120
LR = 1e-3
# EPOCHS = 10
# MODEL_NAME = "gtasa_driving_model-b1_to_b8_test_model.model"
MODEL_NAME = "gtasa_driving_model-b1_to_b8_test_model_round2.model"

t_time = 0.045
t_time_for_fwd = 0.018
AUTO_STRAIGHT = True
straight_frame_diff = 0

def straight():
    # ReleaseKey(A)
    # ReleaseKey(D)
    PressKey(W)

    time.sleep(t_time_for_fwd)

    ReleaseKey(W)

def left():
    global AUTO_STRAIGHT
    # print(straight_frame_diff)
    ReleaseKey(D)
    PressKey(A)
    if AUTO_STRAIGHT:
        PressKey(W)

    time.sleep(t_time)
    
    ReleaseKey(A)
    ReleaseKey(W)

def right():
    global AUTO_STRAIGHT
    # PressKey(W)
    ReleaseKey(A)
    PressKey(D)
    if AUTO_STRAIGHT:
        PressKey(W)
    
    time.sleep(t_time)
    
    ReleaseKey(W)
    ReleaseKey(D)
    
model = alexnet(WIDTH, HEIGHT, LR)
model.load(f"./saved_model/{MODEL_NAME}")

def main():
    global AUTO_STRAIGHT
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False

    frame_number = 0
    while(True):
        
        frame_number += 1

        if not paused:
            # 800x600 windowed mode
            #screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            # FOR GTA-SA
            screen = grab_screen(region=(1, 35, 640, 420))
            # print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))

            prediction = model.predict([screen.reshape(160,120,1)])[0]
            print(prediction)

            # turn_thresh = .75
            turn_thresh = .65
            fwd_thresh = 0.70

            pred_max = np.argmax(prediction)

            if frame_number % 60 == 0:
                AUTO_STRAIGHT = not AUTO_STRAIGHT

            if pred_max == 0:
                # print(straight_frame_diff)
                left()
                print("a")
            elif pred_max == 1:
                straight()
                print("w")
            else:
                right()
                print("d")

                # print("going forward")
            # if prediction[1] > fwd_thresh:
            #     # straight()
            #     print("w")
            # elif prediction[0] > turn_thresh:
            #     left()
            #     print("a")
            # elif prediction[2] > turn_thresh:
            #     right()
            #     print("d")
            # else:
            #     # straight()
            #     print("i have no idea so w")


        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)
        if 'E' in keys:
            break

main()       
