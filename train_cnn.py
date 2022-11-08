import tensorflow as tf
from utils import *
from cnn_models import *
import configs
from keras import layers
from keras import layers, Sequential
from keras.models import Model

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, \
    UpSampling2D, Add

import os, sys, time, tqdm, datetime

from keras.optimizers import SGD, Adam
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import wandb
from wandb.keras import WandbCallback

# wandb.login()

set_seed(configs.seed)

now = datetime.datetime.now()
now = now.strftime('%Y%m%d%H%M%S')

is_gpu = tf.config.list_physical_devices('GPU') is not None
wandb_dir = '/root/autodl-tmp/dl_project12/wandb_logs' if is_gpu else \
    'D:/UPC_Course3/DL/dl_project1/wandb_logs'

# initialize wandb logging to your project
wandb.init(
    job_type='Training',
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
lr = config.learning_rate
weight_decay = config.weight_decay
early_stopping = config.early_stopping
activation = config.activation
augment = config.augment

print('Build Training dataset')
X_train = tf.keras.utils.image_dataset_from_directory(
    configs.train_folder,
    batch_size=config.batch_size,  # batch_size
    # image_size=(img_height, img_width), # resize
    shuffle=True,
    seed=configs.seed
)

print('Build Validation dataset')
X_val = tf.keras.utils.image_dataset_from_directory(
    configs.val_folder,
    batch_size=config.batch_size,  # batch_size
    # image_size=(img_height, img_width), # resize
    shuffle=False,
    seed=configs.seed,
)

AUTOTUNE = tf.data.AUTOTUNE
X_train = X_train.cache().shuffle(2000).prefetch(buffer_size=AUTOTUNE)
X_val = X_val.cache().prefetch(buffer_size=AUTOTUNE)

print('building model')
# Selecting a model from model library
# model = fc_model()

if config.optimizer == 'adam' and config.weight_decay == 0:
    optimizer = Adam(
        lr=config.learning_rate,
        epsilon=config.epsilon,
        amsgrad=config.amsgrad
    )

elif config.optimizer == 'adam' and config.weight_decay != 0:
    optimizer = tfa.optimizers.AdamW(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        epsilon=config.epsilon,
        amsgrad=config.amsgrad,
    )

# SGD + Momentum: Great if LR decayed properly.
elif config.optimizer == "sgd" and config.weight_decay == 0:
    optimizer = SGD(
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        nesterov=config.nesterov
    )

else:
    optimizer = tfa.optimizers.SGDW(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
    )

print('Bulid Model!')
# model = fc_model()
model = cnn6(config)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=optimizer, metrics=["accuracy"])

# Use early stopping
early_callback = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=1, mode='auto',
                               restore_best_weights=True)
wandb_callback = WandbCallback(
                                # save_model=False,
                                # log_weights=True,
                                # log_gradients=True,
                              )
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)
reduce_lr = ReduceLROnPlateau(min_lr=0.00001)

# optimizer = Adam(lr=0.001, epsilon=0.1, amsgrad=True) do we need to change the epsilon?
# step = tf.Variable(0, trainable=False)
# schedule = tf.optimizers.schedules.PiecewiseConstantDecay()

print('Start Training!')
t0 = time.time()
print('training model')

history = model.fit(X_train,
                    validation_data=X_val,
                    epochs=config.epochs,
                    # callbacks=[reduce_lr],
                    # callbacks=[early_callback, wandb_callback],
                    callbacks=[reduce_lr, wandb_callback],  # other callback?
                    )

print('Model trained in {:.1f}min'.format((time.time() - t0) / 60))
print('now is:  ' + now)
# model.save(now + '.h5')

model_path = '/root/autodl-tmp/dl_project1/saved_model_' + now
if not os.path.exists(model_path):
    os.mkdir(model_path)
model.save(model_path + '.h5')

