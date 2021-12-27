import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

IMAGE_SIZE = 224
BATCH_SIZE = 64
PATH_TRAIN = "/content/output/train"
PATH_VAL = "/content/output/val"
PATH_TEST = "/content/output/test"
def define_generator():
  train_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 60,
    horizontal_flip = True,
    
  )
  validation_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 60,
    horizontal_flip = True,
    
  )
  test_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 60,
    horizontal_flip = True,
    
  )

  train_generator = train_generator.flow_from_directory(
        directory = PATH_TRAIN,
        target_size = (IMAGE_SIZE, IMAGE_SIZE),
        batch_size = BATCH_SIZE,
        class_mode = "sparse",
  )
  validation_generator = validation_generator.flow_from_directory(
        directory = PATH_VAL,
        target_size = (IMAGE_SIZE, IMAGE_SIZE),
        batch_size = BATCH_SIZE,
        class_mode = "sparse",
  )
  test_generator = test_generator.flow_from_directory(
        directory = PATH_TEST,
        target_size = (IMAGE_SIZE, IMAGE_SIZE),
        batch_size = BATCH_SIZE,
        class_mode = "sparse",
  )
  return train_generator, validation_generator , test_generator
save_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath='weights.{epoch:02d}-{val_accuracy:.2f}.hdf5',
                                                    monitor='val_accuracy',
                                                    save_best_only=True)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                    min_delta=0, 
                                                    patience=10, 
                                                    mode='auto', 
                                                    baseline=None, 
                                                    restore_best_weights=True)

input_shape = (IMAGE_SIZE,IMAGE_SIZE,3)
n_classes = 10
# Khoi tao model
model1 = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
base_model = model1

# Them cac lop ben tren
x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu', name='fc1')(x)
x = layers.Dense(128, activation='relu', name='fc2')(x)
x = layers.Dense(128, activation='relu', name='fc2a')(x)
x = layers.Dense(128, activation='relu', name='fc3')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu', name='fc4')(x)

predictions = layers.Dense(n_classes, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Dong bang cac lop duoi, chi train lop ben tren minh them vao
for layer in base_model.layers:
    layer.trainable = False

model.summary()

model.compile(
    optimizer='Adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

train_generator, validation_generator , test_generator = define_generator()
EPOCHS = 20
history = model.fit(
    train_generator,
    steps_per_epoch= train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    verbose=1,
    epochs=EPOCHS,
    callbacks= [save_callbacks, early_stopping], # dung khi luu gia tri tot nhat cua val_acc.
)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), accuracy, label='Training Accuracy')
plt.plot(range(EPOCHS), val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Scores is just a list containing loss and accuracy value
scores = model.evaluate(test_generator)
scores

# Save model
model.save("/content/drive/MyDrive/DL/Tomato/tomato.h5")
