import tensorflow as tf
from tensorflow import keras
import configs
import numpy as np
import os
import time
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras
import matplotlib
import matplotlib.pyplot as plt
# import sklearn
from sklearn.metrics import classification_report, confusion_matrix

X_val = tf.keras.utils.image_dataset_from_directory(
    configs.val_folder,
    batch_size=256,  # batch_size
    # image_size=(img_height, img_width), # resize
    shuffle=False,
    seed=configs.seed,
)

X_test = tf.keras.utils.image_dataset_from_directory(
    configs.test_folder,
    batch_size=256,  # batch_size
    # image_size=(img_height, img_width), # resize
    shuffle=False,
    seed=configs.seed,
)

# load the model
model_path = '/root/autodl-tmp/dl_project1/20221024163904_best.h5'
# loaded_model = tf.keras.models.load_model('/tmp/model')


def evaluate_model(model, test_generator, experiment=True):
    """
    This functions creates interesting metrics to check model performance
    :param model:  Model to evaluate
    :param test_generator: Test generator
    :param experiment: Specifies whether ths actual run is an experiment
    :return: It saves the accuracy and loss plots
    """

    # Evaluate the model
    test_generator.reset()
    score = model.evaluate(test_generator, verbose=0)

    if experiment:
        file = open(f'experiments/{exp}/test_info.txt', "a+")
    else:
        file = open(f'model_evaluation.txt', "a+")

    file.write(f"\t - Loss: {str(score[0])} \n \t - Accuracy on test: {str(score[1])}\n")

    # Confusion Matrix (validation subset)
    test_generator.reset()
    pred = model.predict(test_generator, verbose=0)

    # Assign most probable label
    predicted_class_indices = np.argmax(pred, axis=1)

    # Get class labels
    labels = (test_generator.class_indices)
    target_names = labels.keys()

    # Plot statistics
    file.write("\n")
    file.write(classification_report(test_generator.classes, predicted_class_indices, target_names=target_names))
    file.close()


def create_confusion_matrix(model, eval_gen, experiment=True):
    """
    This function evaluates a model. It shows validation loss and accuracy, classification report and confusion matrix.
    :param model: Model to evaluate
    :param eval_gen: Evaluation generator
    :param experiment: Specifies whether ths actual run is an experiment
    """

    # Evaluate the model
    eval_gen.reset()
    score = model.evaluate(eval_gen, verbose=0)
    print('\nLoss:', score[0])
    print('Accuracy:', score[1])

    # Confusion Matrix (validation subset)
    eval_gen.reset()
    pred = model.predict(eval_gen, verbose=0)

    # Assign most probable label
    predicted_class_indices = np.argmax(pred, axis=1)

    # Get class labels
    labels = eval_gen.class_indices
    target_names = labels.keys()

    # Plot statistics
    print(classification_report(eval_gen.classes, predicted_class_indices, target_names=target_names))

    cf_matrix = confusion_matrix(np.array(eval_gen.classes), predicted_class_indices)
    fig, ax = plt.subplots(figsize=(15, 15))
    heatmap = sns.heatmap(cf_matrix, annot=False, cmap='Blues', cbar=True, square=False,
                          xticklabels=target_names, yticklabels=target_names)
    fig = heatmap.get_figure()

    if experiment:
        fig.savefig(f'experiments/{exp}/MAMe_confusion_matrix.pdf')
    else:
        fig.savefig('MAMe_confusion_matrix.pdf')



