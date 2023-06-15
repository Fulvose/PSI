#1
import matplotlib.pyplot as pl
plt.plot(history_sgd.history['loss'], label = "tarina")
plt.plot(history_sgd.history['val_loss'], label = "test")
plt.legend()
plt.show()

RMSprop = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
Adagrad = keras.optimizers.Adagrad(learning_rate=0.01)
Adadelta = keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
Adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
Adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)