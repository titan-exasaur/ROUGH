import tensorflow
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential

x = np.random.random((1000, 20))
y = np.random.randint(2, size=(1000, 1))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x, y, epochs=50, batch_size=32, verbose=2)

import pandas as pd
data = pd.DataFrame(history.history)
data.to_csv("history.csv", index = False)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], color='g')
plt.plot(history.history['loss'], color='r')
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.legend(['Accuracy', 'Loss'], loc='upper left')
plt.show()
