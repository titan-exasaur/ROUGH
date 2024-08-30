import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(422)

train_data_dir = "/media/kumar/HDD1/INFIDATA/INFIDATA PROJECTS/OCT-DEC RBITS PROJECTS/4 [RNSIT] LEAF DISEASE/data/train"
test_data_dir = "/media/kumar/HDD1/INFIDATA/INFIDATA PROJECTS/OCT-DEC RBITS PROJECTS/4 [RNSIT] LEAF DISEASE/data/val"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

train_gen = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    shuffle=True,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224,224),
    batch_size=32,
    shuffle=True,
    class_mode='categorical',
    subset='validation'
)

test_gen = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    shuffle=False,
    class_mode='categorical'
)

#building the CNN model

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=[224,224,3]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

print(model.summary())

from tensorflow.keras.optimizers import Adam
optimser = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimser,
    loss='categorical_crossentropy',
    metrics='accuracy'
)

early_stopping_criteria = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
    verbose=2,
    restore_best_weights=True
)

history = model.fit(
    train_gen,
    epochs=9,
    validation_data=val_gen,
    callbacks=[early_stopping_criteria]
)

model.save("2 ML, DL/leaf_classifier.h5")

scores = model.evaluate(test_gen)
print(f"Test loss : {scores[0]}")
print(f"Test accuracy : {scores[1]}")

training_meta_data = pd.DataFrame(history.history)
training_meta_data.to_csv("2 ML, DL/training_meta_data.csv", index = False)

plt.plot(history.history['accuracy'], color='green', label='Training accuracy')
plt.plot(history.history['loss'], color='red', label='Training loss')
plt.xlabel('epochs')
plt.ylabel('accuracy/loss')
plt.title('Training metrics vs epochs')
plt.legend(['accuracy', 'loss'], loc='top right')
plt.show()