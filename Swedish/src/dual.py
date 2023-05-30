import cv2

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_ranking as tfr
import seaborn as sn
import pandas as pd
import shutil
from skimage.filters import threshold_otsu
from Utils.plots import *
from load_data import *
import sys

class OtsuThresholdLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(OtsuThresholdLayer, self).__init__()

    def build(self, input_shape):
        self.input_dim = input_shape
        print(self.input_dim)

    def otsu_converter(self, inputs):
        gray_images = tf.image.rgb_to_grayscale(inputs)
        binary_images = []

        for gray_image in gray_images:
            threshold = threshold_otsu(tf.squeeze(gray_image))
            binary_image = tf.where(gray_image > threshold, 255, 0)
            binary_images.append(binary_image)

        return tf.image.grayscale_to_rgb(tf.stack(binary_images))
    
    def call(self, inputs):
        # return self.otsu_converter(inputs)
        return tf.reshape(tf.py_function(self.otsu_converter, [inputs], tf.Tensor), shape=self.input_dim) 

class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset):
        super().__init__()
        self.test_dataset = test_dataset

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.test_dataset, verbose=0)
        logs['test_loss'] = loss  
        logs['test_accuracy'] = acc

def sobel_edge(image):
    # gray = tf.image.rgb_to_grayscale(image)
    grad_comp = tf.image.sobel_edges(image)
    grad_mag_comp = grad_comp ** 2
    grad_mag_square = tf.math.reduce_sum(grad_mag_comp, axis=-1)
    grad_mag_img = tf.sqrt(grad_mag_square)
    # res = tf.image.grayscale_to_rgb(grad_mag_img)
    res = grad_mag_img
    return res

def build_model(NUM_CLASSES):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])
    inputs = tf.keras.Input(IMAGE_SHAPE)
    
    # x = tf.keras.layers.ambda(lambda img: sobel_edge(img), name="sobel_edge")(inputs)
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=IMAGE_SHAPE, weights="imagenet")
    
    model.trainable = False
    x = model(x, training=False)
    # rebuild top layers
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool_x")(x)

    y = tf.keras.layers.Lambda(lambda img: sobel_edge(img), name="sobel_edge")(inputs)
    y = data_augmentation(y)
    y = tf.keras.applications.mobilenet_v2.preprocess_input(y)
    y = model(y, training=False)
    # rebuild top layers
    y = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool_y")(y)

    concat = tf.keras.layers.Concatenate(axis=1)([x, y])
    top_dropout_rate = 0.2
    concat = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(concat)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(concat)

    model = tf.keras.Model(inputs, outputs)

    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
    
    model.compile(optimizer=optimizer, 
                  loss="categorical_crossentropy",
                  metrics=['accuracy'],
                  run_eagerly=True)

    return model

def train(model, train_dataset, valid_dataset):
    history = model.fit(train_dataset,
                        epochs=EPOCHS,
                        validation_data=valid_dataset)
    
    return history

if __name__ == "__main__":
    IMAGE_SIZE = (224, 224)
    IMAGE_SHAPE = IMAGE_SIZE + (3,)
    BATCH_SIZE = 32
    EPOCHS = 10
    # basemodel_dataset_extra
    name = "mobilev2_Swedish_dual_sobel"
    train_dir = "../train_processed"

    save_dir = f"../Plots/{name}"
    model_dir = f"../Model/{name}"

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    train_dataset, valid_dataset, test_dataset, CLASS_NAMES = load_data(train_dir=train_dir,
                                                                                    batch_size=BATCH_SIZE,
                                                                                    image_size=IMAGE_SIZE)
    if len(sys.argv) > 1:
        is_train = sys.argv[1]
    else:
        is_train = False

    if is_train:
        model = build_model(len(CLASS_NAMES))
        model.summary()
        history = train(model=model,
                        train_dataset=train_dataset,
                        valid_dataset=valid_dataset)
        evaluate_and_plots(history=history, save_dir=save_dir)
        model.save(model_dir)
    else: 
        model = tf.keras.models.load_model(model_dir)

    print("Evaluate on test dataset")
    loss, acc = model.evaluate(test_dataset)
    print(f"Loss, acc: {loss}, {acc}")