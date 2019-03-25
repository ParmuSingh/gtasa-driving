# create_training_data.py
# created by sentdex, modified by parmusingh
import numpy as np
from sentdex_grabscreen import grab_screen
import cv2
import time
from sentdex_getkeys import key_check
import os


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
    [A,W,D] boolean values.
    '''
    output = [0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output


file_path = "../data/version4/batch1/"
file_name = 'training_data.npy'

file = file_path + file_name
if os.path.isfile(file):
    print('File exists, loading previous data!')
    training_data = list(np.load(file))
else:
    print('File does not exist, starting fresh!')
    training_data = []


def main():

    for i in list(range(6))[::-1]:
        print(i+1)
        time.sleep(1)
    print("data collection started..")

    paused = False
    while(True):

        if not paused:
            # 800x600 windowed mode
            # screen = grab_screen(region=(0,40,800,640))
            game_screen = grab_screen(region=(1, 35, 640, 420))
            last_time = time.time()
            game_screen = cv2.cvtColor(game_screen, cv2.COLOR_BGR2GRAY)
            game_screen = cv2.resize(game_screen, (160,120))
            # resize to something a bit more acceptable for a CNN
            keys = key_check()
            output = keys_to_output(keys)

            training_data.append([game_screen, output])
            
            if len(training_data) % 1000 == 0:
                print(len(training_data))
                np.save(file,training_data)

        keys = key_check()
        # print(keys)
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)
        if 'E' in keys:
            break


main()
print("data collected.")
