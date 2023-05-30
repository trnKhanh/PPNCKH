import tensorflow as tf

def load_data(train_dir, batch_size, image_size):
    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                shuffle=True,
                                                                batch_size=batch_size,
                                                                image_size=image_size,
                                                                label_mode="categorical")
    CLASS_NAMES = train_dataset.class_names
    NUM_CLASSES = len(CLASS_NAMES)

    val_batches = tf.data.experimental.cardinality(train_dataset)
    test_dataset = train_dataset.take(val_batches // 5)
    train_dataset = train_dataset.skip(val_batches // 5)
    


    val_batches = tf.data.experimental.cardinality(train_dataset)
    valid_dataset = train_dataset.take(val_batches // 5)
    train_dataset = train_dataset.skip(val_batches // 5)

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    valid_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, valid_dataset, test_dataset, CLASS_NAMES