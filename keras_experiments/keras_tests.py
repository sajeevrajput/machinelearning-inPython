from keras.models import Sequential
from keras.layers import Dense
import numpy as np

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

X = np.random.random((1000, 100))
y = np.random.randint(2, size=(1000, 1))

model.fit(X, y, batch_size=64, epochs=2, verbose=1)
