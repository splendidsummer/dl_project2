import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from utils import *
from cnn_models import *
import configs
import sys, os, time, datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import wandb

set_seed(configs.seed)
now = datetime.datetime.now()
now = now.strftime('%Y%m%d%H%M%S')


def get_labels(batchset):
    labels = []
    for _, label in batchset:
        labels.append(label)
    print(len(labels))

    labels = tf.concat(labels, axis=0).numpy()
    return labels


def get_class_accuracy(class_label, golden_labels, preds):
    idx = np.argwhere(golden_labels == class_label)
    labels = golden_labels[idx]
    class_preds = preds[idx]

    acc = accuracy_score(labels, class_preds)
    return acc


def get_all_acc(class_names, golden_labels, preds):
    accs = []
    num_labels = len(class_names)
    for class_label in range(num_labels):
        acc = get_class_accuracy(class_label, golden_labels, preds)
        accs.append(acc)
    return accs


def plot_accs(accs):
    plt.figure(figsize=(30, 30))
    plt.plot(range(29), accs, color='green', marker='o', linestyle='dashed',
         linewidth=5, markersize=40)
    plt.xticks(range(0, 29, 1), fontsize=40, rotation=90)
    plt.yticks(fontsize=40)
    plt.show()


def load_model(artifact_url):
    optimizer = tfa.optimizers.AdamW(
        weight_decay=0.0001
    )
    run = wandb.init()
    artifact = run.use_artifact(artifact_url, type='model')
    artifact_dir = artifact.download()
    model = tf.keras.models.load_model(artifact_dir)
    print(model.summary())
    return model


def plot_accs(accs, img_name):
    plt.figure(figsize=(30, 30))
    plt.plot(range(29), accs, color='green', marker='o', linestyle='dashed',
         linewidth=5, markersize=40)
    plt.xticks(range(0, 29, 1), fontsize=40, rotation=90)
    plt.yticks(fontsize=40)
    plt.savefig(img_name + '.jpg')


if __name__ == '__main__':

    artifacts_url = 'unicorn_upc_dl/best_model_sweep_batchsize/model-elated-sweep-2:v16'
    model = load_model(artifacts_url)

    X_val = tf.keras.utils.image_dataset_from_directory(
        configs.val_folder,
        batch_size=256,  # batch_size
        # image_size=(img_height, img_width), # resize
        shuffle=False,
        seed=configs.seed,
    )
    val_labels = get_labels(X_val)

    X_test = tf.keras.utils.image_dataset_from_directory(
        configs.test_folder,
        batch_size=256,  # batch_size
        # image_size=(img_height, img_width), # resize
        shuffle=False,
        seed=configs.seed,)
    test_labels = get_labels(X_test)

    class_names = X_val.class_names
    val_score = model.evaluate(X_val)
    val_preds = model.predict(X_val)
    val_preds = np.argmax(val_preds, axis=1)

    test_score = model.evaluate(X_test)
    test_preds = model.predict(X_test)
    test_preds = np.argmax(test_preds, axis=1)

    print(classification_report(val_labels, val_preds, target_names=class_names))
    print(classification_report(test_labels, test_preds, target_names=class_names))

    val_accs = get_all_acc(class_names, val_labels, val_preds)
    test_accs = get_all_acc(class_names, test_labels, test_preds)

    plot_accs(val_accs, 'validation_accs')
    plot_accs(test_accs, 'test_acc')

