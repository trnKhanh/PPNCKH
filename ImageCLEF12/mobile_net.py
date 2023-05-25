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


IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = IMAGE_SIZE + (3,)
BATCH_SIZE = 32
EPOCHS = 20

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
        return self.otsu_converter(inputs)
        # return tf.reshape(tf.py_function(self.otsu_converter, [inputs], tf.Tensor), shape=self.input_dim) 

class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset):
        super().__init__()
        self.test_dataset = test_dataset

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.test_dataset, verbose=0)
        logs['test_loss'] = loss  
        logs['test_accuracy'] = acc

if __name__ == "__main__":
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

    plot_dir = "./mobilev2"
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    train_dir = "./scan_train"
    test_dir = "./scan_test"

    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMAGE_SIZE,
                                                                label_mode="categorical")
    test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                shuffle=False,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMAGE_SIZE,
                                                                label_mode="categorical")
    file_paths = test_dataset.file_paths 
    
    CLASS_NAMES = train_dataset.class_names
    NUM_CLASSES = len(os.listdir(train_dir))

    val_batches = tf.data.experimental.cardinality(train_dataset)
    valid_dataset = train_dataset.take(val_batches // 5)
    train_dataset = train_dataset.skip(val_batches // 5)


    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    valid_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # otsu = tf.keras.layers.Lambda(lambda img: otsu_converter(img), name="otsu")
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])

    inputs = tf.keras.Input(IMAGE_SHAPE)
    
    x = OtsuThresholdLayer()(inputs)
    x = data_augmentation(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=IMAGE_SHAPE, weights="imagenet")
    
    model.trainable = False
    x = model(x, training=False)

    # rebuild top layers
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)

    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

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

    model.summary()

    history = model.fit(train_dataset,
                        epochs=EPOCHS,
                        validation_data=valid_dataset)

    print("Evaluate on test dataset")
    loss, acc = model.evaluate(test_dataset)
    print(f"Loss, acc: {loss}, {acc}")

    # loss and accuracy plots
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim()
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(f'{plot_dir}/loss_acc_plot.png')

    print("Evaluate on test data")
    results = model.evaluate(test_dataset)
    print("test loss, test acc:", results)

    predictions = []
    labels = []
    wrong = []
    cnt = 0
    for X, Y in test_dataset:
        pred = np.argmax(model.predict(X), axis=1)
        label = np.argmax(Y.numpy(), axis=-1)
        predictions.extend(pred)
        labels.extend(label)

        for i in range(len(pred)):
            if pred[i] != label[i]:
                wrong.append({
                    "image": file_paths[cnt],
                    "pred": CLASS_NAMES[pred[i]],
                    "label": CLASS_NAMES[label[i]],
                })
            cnt = cnt + 1
    
    # get wrong prediction
    for Class in CLASS_NAMES:
        if os.path.isdir(f'{plot_dir}/pred_{Class}'):
            shutil.rmtree(f'{plot_dir}/pred_{Class}')
        os.mkdir(f'{plot_dir}/pred_{Class}')

        img_name = os.listdir(f'{train_dir}/{Class}')[0]
        shutil.copy(f'{train_dir}/{Class}/{img_name}',f'{plot_dir}/pred_{Class}/')

    for item in wrong:
        if not os.path.isdir(f'{plot_dir}/pred_{item["pred"]}/{item["label"]}'):
            os.mkdir(f'{plot_dir}/pred_{item["pred"]}/{item["label"]}')
            
        shutil.copy(item["image"], f'{plot_dir}/pred_{item["pred"]}/{item["label"]}')

    array = tf.math.confusion_matrix(num_classes=NUM_CLASSES,labels=labels, predictions=predictions).numpy()
    df_cm = pd.DataFrame(array, index = [i for i in CLASS_NAMES],
                    columns = [i for i in CLASS_NAMES])
    plt.figure(figsize = (50, 50))
    sn.heatmap(df_cm, annot=False)
    plt.savefig(f'{plot_dir}/confusion_matrix.png', dpi=300)
    