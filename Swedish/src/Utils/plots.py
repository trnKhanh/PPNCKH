import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import os
import shutil 
import tensorflow as tf
from .create_html import create_html

def evaluate_and_plots(history, save_dir):
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
    plt.savefig(f'{save_dir}/loss_acc_plot.png')

def plot_confusion_matrix(model, test_dataset, class_names, save_dir):
    predictions = []
    labels = []

    for X, Y in test_dataset:
        pred = np.argmax(model.predict(X, verbose=False), axis=1)
        label = np.argmax(Y.numpy(), axis=-1)
        predictions.extend(pred)
        labels.extend(label)

    array = tf.math.confusion_matrix(num_classes=len(class_names),labels=labels, predictions=predictions).numpy()
    df_cm = pd.DataFrame(array, index = [i for i in class_names],
                    columns = [i for i in class_names])
    plt.figure(figsize = (50, 50))
    sn.heatmap(df_cm, annot=False)

    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=100)

def visualize_wrong_pred(model, test_dataset, file_paths, train_dir, class_names, save_dir):
    wrong = {}
    cnt = 0
    real_img = {}
    for s in os.scandir(train_dir):
        real_img[s.name] = os.path.join(s.path, os.listdir(s.path)[0])
    wrong_cnt = 0
    for X, Y in test_dataset:
        pred = np.argmax(model.predict(X, verbose=False), axis=1)
        label = np.argmax(Y.numpy(), axis=-1)

        for i in range(len(pred)):
            if pred[i] != label[i]:
                if class_names[pred[i]] not in wrong:
                    wrong[class_names[pred[i]]] = {}
                wrong_cnt += 1
                if class_names[label[i]] not in wrong[class_names[pred[i]]]:
                    wrong[class_names[pred[i]]][class_names[label[i]]] = []
                wrong[class_names[pred[i]]][class_names[label[i]]].append(file_paths[cnt])
            cnt = cnt + 1
    create_html(wrong_pred=wrong,
                real_img=real_img,
                save_dir=save_dir)

