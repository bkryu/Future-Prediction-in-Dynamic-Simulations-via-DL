# Import modules
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
import time
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten
from tensorflow.keras.models import Model

# Remove hidden files from a list of files.
def removeHiddenFiles(list_files):
	counter = 0
	while True:
		if counter < len(list_files):
			if list_files[counter][0] == '.':
				list_files.pop(counter)
			elif len(list_files[counter]) < 7:
				list_files.pop(counter)
			elif list_files[counter][-1] != 'y':
				list_files.pop(counter)
			else:
				counter+= 1
		else:
			break

# Check for correct input
if len(sys.argv) < 3:
	print("Usage:")
	print("  python ./train.py [Training set directory] [Model File Name to Save] {Dev set directory}")
	sys.exit(0)


### Some hyperparameters are defined here
learningRate = 0.005
lambd = 0.1
numEpochs = 60
batchSize = 256
###

print("TensorFlow Version: " + tf.__version__)
print("Κeras Version: " + tf.keras.__version__)
# Save input parameters as variables
path = sys.argv[1]
modelPath = sys.argv[2]
if len(sys.argv)>=4:
	tPath = sys.argv[3]

# Do not show warnings
tf.logging.set_verbosity(tf.logging.ERROR)

### Start reading data files ###
tstart = time.time()

dataPath = path + '/data/'
labelPath = path + '/label/'

print('Directory Path: ' + path)
data_list = os.listdir(dataPath)
label_list = os.listdir(labelPath)

removeHiddenFiles(data_list)
removeHiddenFiles(label_list)

numFiles = len(data_list)
 
for i in range(numFiles):
	data_file= np.load(dataPath + data_list[0])
	label_file = np.load(labelPath + label_list[0])

label_file = np.reshape(label_file,(label_file.shape[0],label_file.shape[1]*label_file.shape[2]))


### If Dev directory was given, read test data files
test_data= None
test_labels = None
if len(sys.argv)==4:
	tDataPath = tPath + 'data/'
	tLabelPath = tPath + 'label/'

	# print('Test File Path: ' + tPath)
	tData_list = os.listdir(tDataPath)
	tLabel_list = os.listdir(tLabelPath)

	removeHiddenFiles(tData_list)
	removeHiddenFiles(tLabel_list)
	numTestFiles = len(tData_list)

	for i in range(numTestFiles):
		test_data = np.load(tDataPath + tData_list[0])
		test_labels = np.load(tLabelPath + tLabel_list[0])

	# test_data = np.reshape(test_data,(numTestFiles, test_data.shape[1]*test_data.shape[2]))
	test_labels = np.reshape(test_labels,(test_labels.shape[0],test_labels.shape[1]*test_labels.shape[2]))


### Models defined here ###
## 1. Naive one hidden fc layer model ##
inputLayer = Input(shape=(8788,3))
fl1 = Flatten()(inputLayer)
fc1 = Dense(5000, activation="relu")(fl1) # Hidden fc layer
outputLayer = Dense(26364, activation="linear")(fc1)

currModel = Model(inputs=inputLayer, outputs=outputLayer)
currModel.compile(optimizer = tf.train.AdamOptimizer(learning_rate = learningRate),
              loss = 'mse',
              metrics=['accuracy', 'mse', 'mae'])

history = currModel.fit(data_file, label_file, epochs=numEpochs, batch_size=batchSize, validation_data=(test_data, test_labels))
hist_df = pd.DataFrame(history.history) 
hist_csv_file = 'FC1_' + 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


modelPathCurr = "FC1_" + modelPath
print("Saving model as: " + modelPathCurr)
currModel.save(modelPathCurr)
print("Clearing current model for next model...")
tf.keras.backend.clear_session()
del currModel, history, hist_df

## 2. Five hidden fc layer model ##
inputLayer = Input(shape=(8788,3))
fl1 = Flatten()(inputLayer)
fc1 = Dense(5000, activation="relu")(fl1)
fc2 = Dense(4000, activation="tanh")(fc1)
fc3 = Dense(3000, activation="tanh")(fc2)
fc4 = Dense(4000, activation="tanh")(fc3)
fc5 = Dense(5000, activation="tanh")(fc4)
outputLayer = Dense(26364, activation="linear")(fc5)

currModel = Model(inputs=inputLayer, outputs=outputLayer)
currModel.compile(optimizer = tf.train.AdamOptimizer(learning_rate = learningRate),
              loss = 'mse',
              metrics=['accuracy', 'mse', 'mae'])

history = currModel.fit(data_file, label_file, epochs=numEpochs, batch_size=batchSize, validation_data=(test_data, test_labels))
hist_df = pd.DataFrame(history.history) 
hist_csv_file = 'FC5_' + 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


modelPathCurr = "FC5_" + modelPath
print("Saving model as: " + modelPathCurr)
currModel.save(modelPathCurr)
print("Clearing current model for next model...")
tf.keras.backend.clear_session()
del currModel, history, hist_df

## 3. One ResNet Identity Block ##
inputLayer = Input(shape=(8788,3))
fl1 = Flatten()(inputLayer)
fc1 = Dense(5000, activation="relu")(fl1)
fc2 = Dense(4000, activation="tanh")(fc1)
fc3 = Dense(4000, activation="tanh")(fc2)
fc4 = Dense(5000, activation="tanh")(fc3)
fcLast = Dense(26364, activation="linear")(fc4)
outputLayer = tf.keras.layers.add([fl1, fcLast])

currModel = Model(inputs=inputLayer, outputs=outputLayer)
currModel.compile(optimizer = tf.train.AdamOptimizer(learning_rate = learningRate),
              loss = 'mse',
              metrics=['accuracy', 'mse', 'mae'])

history = currModel.fit(data_file, label_file, epochs=numEpochs, batch_size=batchSize, validation_data=(test_data, test_labels))
hist_df = pd.DataFrame(history.history) 
hist_csv_file = 'ResNet1_' + 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


modelPathCurr = "ResNet1_" + modelPath
print("Saving model as: " + modelPathCurr)
currModel.save(modelPathCurr)
print("Clearing current model for next model...")
tf.keras.backend.clear_session()
del currModel, history, hist_df

tend = time.time()
print('Elapsed Time: ' + str(tend - tstart))