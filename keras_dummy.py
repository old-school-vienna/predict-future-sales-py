import numpy as np
import tensorflow.python.keras as keras
import tensorflow.python.keras.layers as kerasl

# Create a simple model.
inputs = keras.Input(shape=(32,))
outputs = kerasl.Dense(1)(inputs)
model = keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="mean_squared_error")

x = np.random.random((128, 32))
y = np.random.random((128, 1))
print("-- x type", type(x))
print("-- x shape", x.shape)
print("-- y type", type(y))
print("-- y shape", y.shape)

hist = model.fit(x, y)
print("-- history", type(hist))

y1 = model.predict(x)
print("-- y1:", type(y1))
print("-- y1:", y1.shape)

mse = model.evaluate(x, y)
print("-- evaluated mse", mse)
