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

img_set=[]   # Training Image set
steer=[]     # Corresponding steering angles

#####################################################################
#### Import options for local and Udacity simulator

data_from_udacity=1
csv_path='data/'  # Driving log path
data_path=csv_path+'IMG/'   # Training images path
i_c=0;  i_l=0;   i_r=0;  i_cor=0  # Counters to check no of corrupted images if present
print("\nPreprocessing Training data....")

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

for c,l,r,j,pb in zip(training_data['center'],training_data['left'],training_data['right'],training_data['steering'],
                      tqdm(range(len(training_data['center'])))):
    img_center = cv2.imread(data_path+c) # Reading center camera image
    img_left = cv2.imread(data_path+l)  # Reading left camera image
    img_right = cv2.imread(data_path+r) # Reading right camera image

    ## To check if any of the images are corrupted
    if (img_center is None):
        i_c=i_c+1
        continue
    elif (img_left is None):
        i_l=i_l+1
        continue
    elif (img_right is None):
        i_r=i_r+1
        continue

    i_cor=i_cor+1
    img_c=cv2.cvtColor(img_center,cv2.COLOR_BGR2RGB); img_l=cv2.cvtColor(img_left,cv2.COLOR_BGR2RGB)
    img_r=cv2.cvtColor(img_right,cv2.COLOR_BGR2RGB); img_flip=np.fliplr(img_c) # Flipped image for data augmentation
    img_set.extend([img_c,img_l,img_r,img_flip]) # Set for the training images
    steer.extend([j,j+1,j-1,-j])  # corresponding set for steering angles
print("Preprocessing finished")
print(i_c,i_l,i_r,i_cor)
print("Total no of images available = ", len(img_set))  # Total no images available for training

X,y=shuffle(np.array(img_set),np.array(steer))   # shuffling the image dataset
X_train,X_val,y_train,y_val=train_test_split(X, y, test_size=0.2)  # Splitting the the datset into training and validation
n=len(y_train) # Total no of images in training dataset

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
pred=Dense(1,kernel_initializer='normal')(x)
model = Model(inputs=inp, outputs=pred)  ## Prediction layer
#print(model.summary())  ## Used to print model summary

model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_squared_error'])

# commented for Udacity Gpu as it was slowing down the model training
#train_datagen=t_datagen.flow(X_train,y_train, batch_size=bs)
#val_datagen=v_datagen.flow(X_val,y_val, batch_size=bs)

#model.fit_generator(train_datagen,steps_per_epoch=len(X_train)/(bs),epochs=2,verbose=1,validation_data=val_datagen,validation_steps=len(X_val)/bs)
model.fit(X_train,y_train,batch_size=bs,epochs=3,validation_data=(X_val,y_val))
model.save('model.h5')  # model saving