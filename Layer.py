from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
classifier = Sequential()
classifier.add(Convolution2D(96, 11, strides = (4, 4), padding = &#39;valid&#39;, input_shape=(224, 224, 3), activation =
&#39;relu&#39;))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = &#39;valid&#39;))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(256, 11, strides = (1, 1), padding=&#39;valid&#39;, activation = &#39;relu&#39;))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding=&#39;valid&#39;))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(384, 3, strides = (1, 1), padding=&#39;valid&#39;, activation = &#39;relu&#39;))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(384, 3, strides = (1, 1), padding=&#39;valid&#39;, activation = &#39;relu&#39;))
classifier.add(BatchNormalization())

classifier.add(Convolution2D(256, 3, strides=(1,1), padding=&#39;valid&#39;, activation = &#39;relu&#39;))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = &#39;valid&#39;))
classifier.add(BatchNormalization())
classifier.add(Flatten())
# Full Connection Step
classifier.add(Dense(units = 4096, activation = &#39;relu&#39;))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 4096, activation = &#39;relu&#39;))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization()
classifier.add(Dense(units = 1000, activation = &#39;relu&#39;))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 38, activation = &#39;softmax&#39;))
classifier.summary()
