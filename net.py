  model = Sequential()
    model.add(Convolution2D(16,7,7,input_shape=(30,30,3),dim_ordering='tf', subsample=(1,1)))                                                                
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Convolution2D(16,5,5, subsample=(1,1),border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Convolution2D(16,3,3, subsample=(1,1),border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Convolution2D(16,3,3, subsample=(1,1),border_mode='same'))                                           
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
    model.add(Dense(2))                                                                                                
    model.summary()

    return model
