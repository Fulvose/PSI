keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)



model2 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model2.summary()

sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model2.compile(loss="sparse_categorical_crossentropy",
              optimizer=sgd,
              metrics=["accuracy"])

history2 = model2.fit(X_train, y_train, epochs=30,
    validation_data=(X_valid, y_valid))

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)



model3 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model3.summary()
Adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model3.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam,
              metrics=["accuracy"])
history3 = model3.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)



model4 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model4.summary()
Adam2 = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model4.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam2,
              metrics=["accuracy"])


history4 = model4.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

import pandas as pd

pd.DataFrame(history3.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

pd.DataFrame(history4.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()