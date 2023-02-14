# Original author's code


import numpy as np
import keras
import tensorflow as tf
from multiprocessing import Pool
# from keras.constraints import maxnorm
# from keras import backend as K
# from keras.optimizers import SGD
from keras.utils import np_utils

# from keras.callbacks import ModelCheckpoint
# from keras.models import model_from_json
import json
import threading
import h5py
from sklearn.utils import class_weight

h5_file_location = '/content/drive/MyDrive/AI/final_dataset.hdf5' #enter ur location of dataset
dataset = h5py.File(h5_file_location, 'r')

# Loading training
X_train = dataset[u'X_train']
Y_train = np_utils.to_categorical(dataset[u'Y_v_g_train'],num_classes=541)
print(len(X_train))
print(X_train.shape)
print("training data done!!")

    # Loading Validation
X_val = dataset[u'X_val']
Y_val = np_utils.to_categorical(dataset[u'Y_v_g_val'],num_classes=541)
print("validation data done!!")

 # Generating a Random permutation for Training 
training_random_indexes = np.random.permutation(len(Y_train))
print("random indexes done!!")

    # Batch Size for Training
batch_size_train = 500

    # Number of classes
num_classes = Y_val.shape[1]

    # Epochs for training
epochs = 15

    # Maximum number of iterations for each epoch
max_train_iter_epoch = np.ceil( float(len(Y_train)) / float(batch_size_train))
print(len(Y_train))
print(num_classes)

def train_generator():

    while 1:
        for count_train in range(int(max_train_iter_epoch)):
            if count_train<int(max_train_iter_epoch)-1:
                x_train = X_train[np.sort(training_random_indexes[count_train*batch_size_train:(count_train+1)*batch_size_train]),:]
                y_train = Y_train[np.sort(training_random_indexes[count_train*batch_size_train:(count_train+1)*batch_size_train]),:]
            else :
                x_train = X_train[len(Y_train)-batch_size_train:len(Y_train),:]
                y_train = Y_train[len(Y_train)-batch_size_train:len(Y_train),:]
            x_train = np.divide(np.asarray(x_train,dtype='float32'),255.0)
            yield (x_train,y_train)

tf.keras.backend.set_image_data_format('channels_first')

#MODEL
model = keras.models.Sequential()
model.add(keras.layers.convolutional.Conv2D(20,(3,3),input_shape=(1,32,32),padding='same',activation='relu'))
model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(keras.layers.convolutional.Conv2D(50,(3,3),activation='relu',padding='same'))
model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(keras.layers.convolutional.Conv2D(100,(3,3),activation='relu',padding='same'))
model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(500,activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(num_classes,activation='softmax'))

opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
model.summary()

model.fit( train_generator(),  validation_data=(np.asarray(X_val,dtype=np.float32)/255.0, Y_val),validation_steps=5,
          verbose=0,
          steps_per_epoch=int(max_train_iter_epoch),
          epochs=10, 
          use_multiprocessing=False)

json_txt = model.to_json() #saving the model
with open('model_v_g.json','w') as outfile:
  json.dump(json_txt,outfile)
  outfile.close()

model.save_weights("model_v_g.hdf5")
