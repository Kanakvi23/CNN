
import os
import numpy as np
import h5py
os.environ["KERAS_BACKEND"]="tensorflow"
from keras.models import Sequential, Model
from keras.layers import Input, Merge
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.optimizers import Adam, Adadelta, SGD, Adagrad, RMSprop
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.initializers import RandomNormal
from keras.layers.normalization import BatchNormalization
import scipy
import my_generate
import utils_akshay
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from model import get_unet
import random


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

X_train, y_train, XF_train = my_generate.get_data("supernova", shuffle=True, fold=0)
X_temp = np.zeros((X_train.shape[0],30,30,3))
for i in range(X_train.shape[0]):
    print X_temp[i,:,:,0].shape
    print X_train[i,0,:,:].shape
    X_temp[i,:,:,0]=(X_train[i,0,9:39,9:39])
    X_temp[i,:,:,1]=(X_train[i,1,9:39,9:39])
    X_temp[i,:,:,2]=(X_train[i,2,9:39,9:39])
X_train = X_temp
print "X_train"
print X_train.shape
print "y_train"
print y_train.shape


unique, counts = np.unique(y_train, return_counts=True)
print dict(zip(unique, counts))

X_test, y_test, XF_test = my_generate.get_data("supernova", shuffle=False, fold=1)
X_temp = np.zeros((X_test.shape[0],30,30,3))
for i in range(X_test.shape[0]):
    X_temp[i,:,:,0]=(X_test[i,0,9:39,9:39])
    X_temp[i,:,:,1]=(X_test[i,1,9:39,9:39])
    X_temp[i,:,:,2]=(X_test[i,2,9:39,9:39])

X_test = X_temp

print X_train.shape
print X_test.shape

idx_class_0 = np.where(y_train==0)[0]
print idx_class_0.shape
idx_class_1 = np.where(y_train==1)[0]
print idx_class_1.shape
numSamples_req = idx_class_0.shape[0]-idx_class_1.shape[0]

X_train_gen = np.zeros((numSamples_req,30,30,3))

j = 0
for i in range(numSamples_req):
    rot = np.random.uniform(-30,30)
    #rot = np.random.choice([0, 90, 180, 270])
    print rot
    idx = np.random.randint(0,idx_class_1.shape[0],size=1)
    iTemp = np.squeeze(X_train[idx_class_1[idx],:])
    iTemp = scipy.ndimage.rotate(iTemp,rot,reshape=False)
    num = np.random.randint(0,3)
    if(num==0):
        np.fliplr(iTemp)
    if(num==2):
        np.flipud(iTemp)
    X_train_gen[i,:,:,:] = iTemp
    if j%50 == 0:
        scipy.misc.imsave((str(i) + "rotated" + ".png"), iTemp)
        scipy.misc.imsave((str(i) + "original"+ ".png"), np.squeeze(X_train[idx_class_1[idx],:]))
    j=j+1

X_train = np.concatenate((X_train,X_train_gen))
print y_train.shape
y_train_temp = np.ones((numSamples_req,))
print y_train_temp.shape
y_train = np.concatenate((y_train,y_train_temp))
#y_train = np_utils.to_categorical(y_train,3)

#datagen = ImageDataGenerator(
#    rotation_range=90,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    horizontal_flip=True,
#    vertical_flip=True)

#datagen.fit(X_train)


    
def discriminator():
    model = Sequential()
    model.add(Convolution2D(16,7,7,input_shape=(30,30,3),dim_ordering='tf', subsample=(1,1)))
    #model.add(LeakyReLU(0.2))
    #model.add(Convolution2D(16,3,3, subsample=(1,1),border_mode='same'))
    #model.add(LeakyReLU(0.2))
    #model.add(Convolution2D(16,3,3, subsample=(1,1),border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Convolution2D(16,5,5, subsample=(1,1),border_mode='same'))
    #model.add(LeakyReLU(0.2))
    #model.add(Convolution2D(16,3,3, subsample=(1,1),border_mode='same'))
    #model.add(LeakyReLU(0.2))
    #model.add(Convolution2D(16,3,3, subsample=(1,1),border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Convolution2D(16,3,3, subsample=(1,1),border_mode='same'))
    #model.add(LeakyReLU(0.2))
    #model.add(Convolution2D(16,3,3, subsample=(1,1),border_mode='same'))
    #model.add(LeakyReLU(0.2))
    #model.add(Convolution2D(16,3,3, subsample=(1,1),border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1000))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1000))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.5))
    #model.add(Dense(2))                                                                                                                      
    #model.add(Activation('softmax'))                                                                                                    
    model.summary()

    return model
    

model = Sequential()
model1 = discriminator()
model.add(model1)
model.add(Dense(2))                                                                                                                                                                                    
model.add(Activation('softmax'))  
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0002),metrics=['accuracy'])


mean = np.mean(X_train)
sd = np.std(X_train)

X_train = (X_train-mean)/sd
X_test = (X_test-mean)/sd

print np.unique(y_train)

y_train=np_utils.to_categorical(y_train,2)

#rotations = [90, 95, 100, 180, 185, 190, 270, 275, 280]

#model.load_weights('astrov_onlycnndeep0.1.h5')


#model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
#                    steps_per_epoch=len(X_train) / 128, epochs=1000)

model.fit(X_train, y_train, nb_epoch=5000, batch_size=128, shuffle=True)

#if i%1==0:

model.save_weights('astrov_onlycnn_aug.h5')
#model1.save_weights('partModel.h5')

        
y_predict = model.predict(X_test,batch_size=32)
y_predict = np.argmax(y_predict,1)

print y_predict
print y_predict.shape
print y_test.shape
#y_test = np.argmax(y_test,1)                                                                                                         
name = 'report/confusion_onlycnn_aug.png'
utils.compute_confusion(y_test,y_predict,name,["bogus", "real"])



