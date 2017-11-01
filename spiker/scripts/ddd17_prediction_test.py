"""DDD17 Prediction from Pretrained Model.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

import numpy as np
import matplotlib.pyplot as plt

import spiker
from spiker.models import utils
from spiker.data import ddd17

# load and process data
data_path = os.path.join(spiker.SPIKER_DATA, "ddd17",
                         "highway-down-1-export.hdf5")
frames, steering = ddd17.prepare_train_data(data_path)
frames = frames[50:-350]/255.
frames -= np.mean(frames, keepdims=True)
steering = steering[50:-350]
num_samples = frames.shape[0]
num_train = int(num_samples*0.7)
#  X_train = frames[:num_train]
#  Y_train = steering[:num_train]
X_test = frames[num_train:]
Y_test = steering[num_train:]

# load model
model_path = os.path.join(
    spiker.SPIKER_EXPS, "resnet-steering-3-5",
    "resnet-steering-3-5-104-0.01.hdf5")
model = utils.load_keras_model(model_path)

# generate prediction
num_test = X_test.shape[0]

Y_predict = np.array([])

for batch in xrange(num_test // 64):
    y_predict = model.predict(X_test[batch*64:(batch+1)*64])
    Y_predict = y_predict if Y_predict.size == 0 else \
        np.vstack((Y_predict, y_predict))
    print ("[MESSAGE] Processed %d/%d, size: %d"
           % (batch+1, num_test//64, Y_predict.shape[0]))

y_predict = model.predict(X_test[-(num_test % 64):])
Y_predict = np.vstack((Y_predict, y_predict))

print (Y_predict.shape)
print (Y_test.shape)

x_axis = np.arange(num_test)

plt.figure()
plt.plot(x_axis, Y_test, "r", label="ground truth")
plt.plot(x_axis, Y_predict, "g", label="predicted")
plt.legend()
plt.show()
