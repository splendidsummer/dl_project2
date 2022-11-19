import tensorflow as tf
from models import *
import os, sys, time, tqdm, datetime
import keras
from keras import layers, optimizers
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import wandb
from wandb.keras import WandbCallback
from models import *
import configs
from utils import *
import tensorflow_addons as tfa
import sklearn


set_seed(configs.seed)

now = datetime.datetime.now()
now = now.strftime('%Y%m%d%H%M%S')

is_gpu = tf.config.list_physical_devices('GPU') is not None
wandb_dir = '/root/autodl-tmp/dl_project2/wandb_logs' if is_gpu else \
    'D:/UPC_Course3/DL/dl_project1/wandb_logs'

# initialize wandb logging to your project
wandb.init(
    job_type='FineTune',
    project=configs.PROJECT_NAME,
    dir=wandb_dir,
    entity=configs.TEAM_NAME,
    config=configs.wandb_config,
    sync_tensorboard=True,
    name='cnn6' + now,
    notes='min_lr=0.00001',
    ####
)

config = wandb.config
batch_size = config.batch_size
first_stage_epochs = config.first_stage_epochs
finetune_epochs = config.finetune_epochs
lr = config.learning_rate
weight_decay = config.weight_decay
early_stopping = config.early_stopping
activation = config.activation
augment = config.augment

print('Build Training dataset')
X_train = tf.keras.utils.image_dataset_from_directory(
    configs.TRAIN_FOLDER,
    batch_size=config.batch_size,  # batch_size
    # image_size=(img_height, img_width), # resize
    shuffle=True,
    seed=configs.seed
)

print('Build Validation dataset')
X_val = tf.keras.utils.image_dataset_from_directory(
    configs.VAL_FOLDER,
    batch_size=config.batch_size,  # batch_size
    # image_size=(img_height, img_width), # resize
    shuffle=False,
    seed=configs.seed,
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = X_train.prefetch(buffer_size=AUTOTUNE)
test_dataset = X_val.prefetch(buffer_size=AUTOTUNE)

print('building model')
# Selecting a model from meodel library
extractor = build_efficientnetb7_extractor()

for data_batch, label_batch in train_dataset:
    out = extractor(data_batch)

    print(111)

