import os
import numpy as np
from alexnet import alexnet

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 15

MODEL_NAME = "gtasa_driving_model-b1_to_b8_test_model.model"
MODEL_PATH = f"./saved_model/{MODEL_NAME}"
model = alexnet(WIDTH, HEIGHT, LR)

if os.path.isfile(f"{MODEL_PATH}.index"):
    print('File exists, loading previous model!')
    model.load(MODEL_PATH)
else:
    print('Model does not exist, starting fresh!')

train_data = np.load("../../data/version4/merge_b1_to_b8_training_data.npy")

test_data = train_data[len(train_data) - 5000:]
train_data = train_data[:len(train_data) -  5000]
	
X = np.array([i[0] for i in train_data]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[1] for i in train_data]

test_X = np.array([i[0] for i in test_data]).reshape(-1, WIDTH, HEIGHT, 1)
test_Y = [i[1] for i in test_data]

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_X}, {'targets': test_Y}),
	snapshot_step = 500, show_metric =True, run_id = MODEL_NAME, batch_size=1024)

# tensorboard --logdir=foo:E/workspace_py/gtasa_driving/model4/v0.2/log

# model.save(MODEL_PATH)
model.save("./saved_model/gtasa_driving_model-b1_to_b8_test_model_round2.model")
print("model saved")