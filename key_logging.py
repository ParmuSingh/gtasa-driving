from pynput.keyboard import Key, Controller,Listener
import time
import numpy as np
from PIL import Image, ImageGrab
import keyboard as keyy
import pickle

# ############################# keyboard ctrl ############################
keyboard = Controller()
actions=[]

for i in list(range(4))[::-1]:
	print(i+1)
	time.sleep(1)
i=1
while i<=1000:

	#############################################
	################# read image ################
	#############################################
	game_screen = ImageGrab.grab(bbox=(1, 35, 640, 420))
	#############################################

	keys = keyy.read_key()
	if 'a' in keys:
		output = 1
		# print('a')
	elif 'd' in keys:
		output = 3
		# print('d')
	elif 'w' in keys:
		output = 0
		# print('w')
	else:
		output = 2
		# print('s')

	# print(f"{i} - {output}")
	actions.append([i, output])

	game_screen.save(f"./data/raw_images/gtasa_frame{i}.png")
	i=i+1

print("1000 frames completed.")
pickle.dump(actions, open("./data/raw_actions/raw_actions.pkl", "wb"))
