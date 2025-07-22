import random
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import regularizers
import tensorflow_model_optimization as tfmot


if __name__ == '__main__':
    # Loading CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    
    # Preprocess the data
    x_train_normalized = x_train.astype('float32')
    x_test_normalized = x_test.astype('float32')
    x_val_normalized = x_val.astype('float32')
    x_train_normalized /= 255
    x_test_normalized /= 255
    x_val_normalized /= 255
    
    # Convert class vectors to binary class matrices
    y_train_normalized = keras.utils.to_categorical(y_train, 10)
    y_test_normalized = keras.utils.to_categorical(y_test, 10)
    y_val_normalized = keras.utils.to_categorical(y_val, 10)
    
    # Set up data augmentation
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode='nearest')
    
    minnet = keras.Sequential()
    minnet.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    minnet.add(layers.MaxPooling2D(pool_size=(2, 2)))
    minnet.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    minnet.add(layers.MaxPooling2D(pool_size=(2, 2)))
    minnet.add(layers.Flatten())
    minnet.add(layers.Dense(64, activation='relu'))
    minnet.add(layers.Dense(10, activation='softmax'))
    
    # Compile the model
    minnet.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    minnet.summary()
    history = minnet.fit(datagen.flow(x_train_normalized, y_train_normalized, batch_size=64), steps_per_epoch=x_train_normalized.shape[0] // 64, epochs=200, validation_data=(x_val_normalized, y_val_normalized))
    test_loss, test_acc = minnet.evaluate(x_test_normalized, y_test_normalized, verbose=2)
    minnet.save("MinNet.h5")
    
    quantize_model = tfmot.quantization.keras.quantize_model
    quantized_model = quantize_model(minnet)
    quantized_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics = ['accuracy'])
    quantized_model.summary()
    history_quantized = quantized_model.fit(datagen.flow(x_train_a, y_train_a, batch_size=64), steps_per_epoch=x_train_a.shape[0] // 64, epochs=200, validation_data=(x_val_a, y_val_a))
    quant_test_loss, quant_test_acc = quantized_model.evaluate(x_test_a, y_test_a, verbose=2)
    quantized_model.save("QuantizedawareMinNet.h5")
    #
    converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_model = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()   
    quant_file = 'MinNet-quantized.tflite'
    with open(quant_file, 'wb') as f:
        f.write(quantized_tflite_model)

    print("done")
