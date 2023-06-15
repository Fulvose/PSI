#1
from keras.models import Sequential
from keras.layers import Dense

from keras.callbacks import History

history = History()
model = Sequential()
model.add(Dense(100,activation="sigmoid",input_shape=(X_train.shape[1],)))
model.add(Dense(50,activation="sigmoid"))
model.add(Dense(10,activation="sigmoid"))
model.add(Dense(1,activation="sigmoid"))
model.summary()

#2
model.compile(loss="binary_crossentropy",optimizer="Adam", metrics=["accuracy"])
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
history = model.fit(X_train, y_train, validation_data= (X_test, y_test), batch_size=32,epochs=100, callbacks=[early_stopping])

import tensorflow as ts
tf.reshape(X_train, [10, 10])

history = model.fit(X_train,
         y_train,
         batch_size=64,
         epochs=10,
         validation_data=(X_test, y_test))