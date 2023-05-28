import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_ranking as tfr
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':

    plot_dir = "./mobilev2"
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    train_dir = "./train_processed"

    EPOCHS = 20
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)
    NUM_CLASSES = len(os.listdir(train_dir))

    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, 
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE,
                                                                label_mode='categorical',
                                                                )

    CLASS_NAMES = train_dataset.class_names
    val_batches = tf.data.experimental.cardinality(train_dataset)
    validation_dataset = train_dataset.take(val_batches // 5)
    train_dataset = train_dataset.skip(val_batches // 5)

    val_batches = tf.data.experimental.cardinality(train_dataset)
    test_dataset = train_dataset.take(val_batches // 8)
    train_dataset = train_dataset.skip(val_batches // 8)
    
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
    prediction_layer = tf.keras.layers.Dense(NUM_CLASSES)
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

    history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=validation_dataset)
    
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
    for X, Y in test_dataset:
        predictions.extend(np.argmax(model.predict(X.numpy()), axis=1))
        labels.extend(np.argmax(Y.numpy(), axis=-1))

    array = tf.math.confusion_matrix(num_classes=NUM_CLASSES,labels=labels, predictions=predictions).numpy()
    df_cm = pd.DataFrame(array, index = [i for i in CLASS_NAMES],
                    columns = [i for i in CLASS_NAMES])
    plt.figure(figsize = (50, 50))
    sn.heatmap(df_cm, annot=False)
    plt.savefig(f'{plot_dir}/confusion_matrix.png', dpi=100)

