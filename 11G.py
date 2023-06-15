from keras.models import Sequential
from keras.layers import Dense

from keras.callbacks import History

history_sgd_adam1 = History()
model2 = Sequential()
model2.add(Dense(100,activation="sigmoid",input_shape=(X_train.shape[1],)))
model2.add(Dense(50,activation="sigmoid"))
model2.add(Dense(10,activation="sigmoid"))
model2.add(Dense(1,activation="sigmoid"))
model2.summary()
sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model2.compile(loss="binary_crossentropy",optimizer=sgd, metrics=["accuracy"])
model2.fit(X_train, y_train, validation_data= (X_test, y_test), batch_size=32,epochs=100, callbacks=[history_sgd_adam1])

from keras.callbacks import LearningRateScheduler
# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
    return lrate

history_adam2 = History()
model3 = Sequential()
model3.add(Dense(100,activation="sigmoid",input_shape=(X_train.shape[1],)))
model3.add(Dense(50,activation="sigmoid"))
model3.add(Dense(10,activation="sigmoid"))
model3.add(Dense(1,activation="sigmoid"))
model3.summary()

sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model3.compile(loss="binary_crossentropy",optimizer=sgd, metrics=["accuracy"])

lrate3 = LearningRateScheduler(step_decay)
model3.fit(X_train, y_train, validation_data= (X_test, y_test), batch_size=32,epochs=100, callbacks=[lrate, history_adam2])

from keras.callbacks import LearningRateScheduler
# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
    return lrate

history_adam3 = History()
model3 = Sequential()
model3.add(Dense(100,activation="sigmoid",input_shape=(X_train.shape[1],)))
model3.add(Dense(50,activation="sigmoid"))
model3.add(Dense(10,activation="sigmoid"))
model3.add(Dense(1,activation="sigmoid"))
model3.summary()

sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model3.compile(loss="binary_crossentropy",optimizer=sgd, metrics=["accuracy"])

lrate3 = LearningRateScheduler(step_decay)
model3.fit(X_train, y_train, validation_data= (X_test, y_test), batch_size=32,epochs=100, callbacks=[lrate, history_adam3])

import matplotlib.pyplot as pl
plt.plot(history_adam3.history['accuracy'], label = "AdamLR3")
plt.plot(history_adam3.history['val_accuracy'], label = "test LR3")

plt.plot(history_adam2.history['accuracy'], label = "AdamLR2")
plt.plot(history_adam2.history['val_accuracy'], label = "test LR2")

plt.plot(history_sgd_adam1.history['accuracy'], label = "Adam")
plt.plot(history_sgd_adam1.history['val_accuracy'], label = "test")
plt.legend()
plt.show()