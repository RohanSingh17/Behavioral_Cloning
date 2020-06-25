#####################################################################
##########    Project 4 -- Behavioral Cloning    ####################
#####################################################################

####################################################################
##Importing all the required libraries

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm  # Used to display the progress bar

###################################################################
#### Keras ImageDataGenerator
# commented as it was slowing the program in GPU mode

# from keras.preprocessing import image
# t_datagen = image.ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2)
# v_datagen = image.ImageDataGenerator(rescale=1./255)

bs=32 # Batch size

########################################################################################
### Python Image data generator
### used for generating dataset in batches
### (It is used to counter above problem with gpu)


## function recives following inputs
## data_file ---- dataframe storing file locations
## data_path ---- folder location where images are stored
## batch_size --- length of batch size to be used for model training

def gen_data(data_file, data_path, batch_size):
    shuffle(data_file) # shuffling dataset
    num = len(data_file)
    while True:
        for offset in range(0, num, batch_size):
            batch=data_file[offset:(offset+batch_size)]  # Batch generation

            img_set=[]  # list for storing batch of training images
            steer=[]    # list for storing batch of corresponding steering angles

            for i in range(len(batch)):
                img_c = cv2.cvtColor(cv2.imread(data_path + batch.iloc[i]['center']), cv2.COLOR_BGR2RGB) # Reading center camera image
                img_l = cv2.cvtColor(cv2.imread(data_path + batch.iloc[i]['left']), cv2.COLOR_BGR2RGB)  # Reading left camera image
                img_r = cv2.cvtColor(cv2.imread(data_path + batch.iloc[i]['right']), cv2.COLOR_BGR2RGB)  # Reading center camera image
                j = batch.iloc[i]['steering']
                img_flip = np.fliplr(img_c)  # Flipping center image

                corr = 1
                img_set.extend([img_c, img_l, img_r, img_flip])
                steer.extend([j, j + corr, j - corr, -j])

            img_set = np.array(img_set)
            steer = np.array(steer)

            yield img_set,steer

#####################################################################
#### Import options for local and Udacity simulator

data_from_udacity=1
csv_path='data/'  # Driving log path
data_path=csv_path+'IMG/'   # Training images path

## Read dataset parameters from driving_log.csv for steering angles and training image locations
if data_from_udacity==0:
    training_data=pd.read_csv(csv_path+'driving_log.csv',names=['center','left','right','steering','throttle','brake','speed'])
    training_data['center']=training_data['center'].apply(lambda x: x.split('\\')[-1])
    training_data['right']=training_data['right'].apply(lambda x: x.split('\\')[-1])
    training_data['left']=training_data['left'].apply(lambda x: x.split('\\')[-1])
else:
    training_data=pd.read_csv(csv_path+'driving_log.csv')
    training_data['center']=training_data['center'].apply(lambda x: x.split('/')[-1])
    training_data['right']=training_data['right'].apply(lambda x: x.split('/')[-1])
    training_data['left']=training_data['left'].apply(lambda x: x.split('/')[-1])

## Importing keras libraries for model architecture

from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, Cropping2D,
from keras import optimizers,Input

##########################################################################
#### Model Architecture

inp=Input(shape=(160,320,3))  # Input layer
x=Cropping2D(cropping=((50,20), (0,0)))(inp)  # Cropping irrelevant portion of the Image
x=Lambda(lambda x: (x/255.0)-0.5)(x)    # Images normalization

## 1st convolutional block of layers
x=Convolution2D(10,(5,5),activation='relu',kernel_initializer='normal')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Convolution2D(5,(1,1),activation='relu',kernel_initializer='normal')(x)

## 2nd convolutional block of layers
x=Convolution2D(30,(5,5),activation='relu',kernel_initializer='normal')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Convolution2D(15,(1,1),activation='relu',kernel_initializer='normal')(x)

## 3rd convolutional block of layers
x=Convolution2D(150,(5,5),activation='relu',kernel_initializer='normal')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Convolution2D(75,(1,1),activation='relu',kernel_initializer='normal')(x)

## Fully connected layers
x=Flatten()(x)
x=Dense(128,kernel_initializer='normal',activation='relu')(x)
x=Dropout(0.5)(x)
pred=Dense(1,kernel_initializer='normal')(x)
model = Model(inputs=inp, outputs=pred)  ## Prediction layer
#print(model.summary())  ## Used to print model summary

model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_squared_error'])

# dataset generation in batches using python generator
train_datagen=gen_data(train_data,data_path, batch_size=bs)
val_datagen=gen_data(validation_data,data_path, batch_size=bs)

model.fit_generator(train_datagen,steps_per_epoch=(len(train_data)*4)//bs,epochs=3,verbose=1,validation_data=val_datagen,validation_steps=(len(validation_data)*4)/bs)
#model.fit(X_train,y_train,batch_size=bs,epochs=3,validation_data=(X_val,y_val))
model.save('model.h5')  # model saving