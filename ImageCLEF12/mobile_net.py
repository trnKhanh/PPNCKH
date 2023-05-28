import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_ranking as tfr
import seaborn as sn
import pandas as pd


if __name__ == "__main__":
    plot_dir = "./mobilev2"
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    IMAGE_SIZE = (224, 224)
    IMAGE_SHAPE = IMAGE_SIZE + (3,)
    BATCH_SIZE = 32
    EPOCHS = 20
    train_dir = "./oversampled_train"
    test_dir = "./oversampled_test"

    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMAGE_SIZE,
                                                                label_mode="categorical")
    test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMAGE_SIZE,
                                                                label_mode="categorical")
    CLASS_NAMES = train_dataset.class_names
    NUM_CLASSES = len(os.listdir(train_dir))

    val_batches = tf.data.experimental.cardinality(train_dataset)
    valid_dataset = train_dataset.take(val_batches // 5)
    train_dataset = train_dataset.skip(val_batches // 5)

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    valid_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])

    inputs = tf.keras.Input(IMAGE_SHAPE)
    x = data_augmentation(inputs)
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
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    # )

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=initial_learning_rate)
    
    model.compile(optimizer=optimizer, 
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    model.summary()
    
    history = model.fit(train_dataset,
                        epochs=EPOCHS,
                        validation_data=valid_dataset)

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
    plt.savefig(f'{plot_dir}/confusion_matrix.png', dpi=300)
    