import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_ranking as tfr


if __name__ == '__main__':

    train_dir = "./train"

    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)
    CLASS_NUM = len(os.listdir(train_dir))

    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, 
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE,
                                                                label_mode='categorical',
                                                                )

    val_batches = tf.data.experimental.cardinality(train_dataset)
    validation_dataset = train_dataset.take(val_batches // 5)
    train_dataset = train_dataset.skip(val_batches // 5)

    
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    # normalized_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    # image_batch, labels_batch = next(iter(normalized_dataset))

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(CLASS_NUM)
    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)

    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = normalization_layer(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    
    model.summary()
    initial_epochs = 20

    history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)
