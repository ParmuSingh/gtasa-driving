import numpy as np
import cv2
import pandas as pd
from collections import Counter
from random import shuffle

training_data = np.load("./training_data.npy")

df = pd.DataFrame(training_data)

print(f"Before: {Counter(df[1].apply(str))}")

lefts = []
rights = []
forwards = []

shuffle(training_data)

for data in training_data:
	img = data[0]
	choice = data[1]

	if choice == [1, 0, 0]:
		lefts.append([img, choice])
	elif choice == [0, 1, 0]:
		forwards.append([img, choice])
	elif choice == [0, 0, 1]:
		rights.append([img, choice])
	else:
		print("no matches")


forwards = forwards[ : len(lefts)][: len(rights)]
lefts = lefts[: len(forwards)]
rights = rights[: len(forwards)]

final_data = forwards + lefts + rights

shuffle(final_data)

print(f"before: {len(training_data)}")
print(f"after: {len(final_data)}")

np.save("./final_training_data.npy", final_data)