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
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.regularizers import l2

def residual_block(inputs, filters, stride, kernel_size=3, regularizer=l2(1e-4)):
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same', kernel_regularizer=regularizer)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=kernel_size, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)

    shortcut = inputs
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='valid', kernel_regularizer=regularizer)(inputs)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def resnet8(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(16, kernel_size=3, padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    num_blocks_list = [1, 1, 1]
    filters_list = [32, 64, 128]

    for i, (num_blocks, filters) in enumerate(zip(num_blocks_list, filters_list)):
        stride = 2 if i > 0 else 1
        for j in range(num_blocks):
            x = residual_block(x, filters, stride if j == 0 else 1)

    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model


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

    classes =['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    model = resnet8(input_shape=(32, 32, 3), num_classes=10)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(datagen.flow(x_train_normalized, y_train_normalized, batch_size=64), steps_per_epoch=x_train_normalized.shape[0] // 64, epochs=200, validation_data=(x_val_normalized, y_val_normalized))
    test_loss, test_acc = model.evaluate(x_test_normalized, y_test_normalized, verbose=2)
    model.save("ResNet8.h5")

    quantize_model = tfmot.quantization.keras.quantize_model
    quantized_model = quantize_model(model)
    quantized_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics = ['accuracy'])
    quantized_model.summary()
    history_quantized = quantized_model.fit(datagen.flow(x_train_normalized, y_train_normalized, batch_size=64),steps_per_epoch=x_train_normalized.shape[0] // 64, epochs=200,validation_data=(x_val_normalized, y_val_normalized))
    quant_test_loss, quant_test_acc = quantized_model.evaluate(x_test_normalized, y_test_normalized, verbose=2)
    quantized_model.save("QuantizedawareResNet8.h5")

    converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_model = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()
    quant_file = 'ResNet8-quantized.tflite'
    acc = evaluate_quantized_model(interpreter, x_test_normalized[0:100], y_test_normalized[0:100])
    print(acc)
    with open(quant_file, 'wb') as f:
        f.write(quantized_tflite_model)
