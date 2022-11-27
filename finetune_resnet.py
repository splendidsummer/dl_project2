import tensorflow as tf
from models import *
import os, sys, time, tqdm, datetime
import math
import keras
from keras import layers, optimizers
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import wandb
from wandb.keras import WandbCallback
from models import *
import configs
from utils import *
import tensorflow_addons as tfa

set_seed(configs.seed)

now = datetime.datetime.now()
now = now.strftime('%Y%m%d%H%M%S')

is_gpu = tf.config.list_physical_devices('GPU') is not None
wandb_dir = '/root/autodl-tmp/dl_project2/wandb_logs' if is_gpu else \
    'D:/UPC_Course3/DL/dl_project1/wandb_logs'

# initialize wandb logging to your project
wandb.init(
    job_type='FineTune_Resnet',
    project=configs.PROJECT_NAME,
    dir=wandb_dir,
    entity=configs.TEAM_NAME,
    config=configs.wandb_config,
    # sync_tensorboard=True,
    name='freeze_all_blocks' + now,
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
n_steps = math.ceil(len(X_train) / batch_size) * 10  # here we set decay_epochs = 10
n_steps_per_epoch = math.ceil(len(X_train) / batch_size)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = X_train.prefetch(buffer_size=AUTOTUNE)
test_dataset = X_val.prefetch(buffer_size=AUTOTUNE)

print('building model')
# Selecting a model from meodel library
model = build_resnet50()

print('Trainable variables in model: ', len(model.trainable_variables))


# Use early stopping
early_callback = EarlyStopping(monitor='val_loss',
                               min_delta=1e-4,
                               patience=10,
                               verbose=1, mode='auto',
                               restore_best_weights=True)

wandb_callback = WandbCallback(
                                save_model=False,
                                log_weights=True,
                                # log_gradients=True,
                              )

reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)

"""
需要edit不同的learning rate 
"""
# There are multiple choice for learning rate scheduler
# Including ReduceLROnPlateau,
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

lr_scheduler = None

if config.lr_scheduler == 'exponential':
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=n_steps*10, decay_rate=0.1)
elif config.lr_scheduler == 'piecewise':
    lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[10. * n_steps_per_epoch, 20. * n_steps_per_epoch],
    values=[0.001, 0.0005, 0.0001])

if lr_scheduler is not None:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer=optimizer, metrics=["accuracy"])

print('Start Training Classifier!')
t0 = time.time()

history = model.fit(X_train,
                    validation_data=X_val,
                    epochs=first_stage_epochs,
                    # callbacks=[reduce_lr_callback],
                    callbacks=[early_callback, wandb_callback],
                    # callbacks = [wandb_callback],
                    # callbacks=[reduce_lr, wandb_callback],
                    )

optimizers = [
    tf.keras.optimizers.Adam(learning_rate=lr),
    tf.kieras.optimizers.Adam(learning_rate=lr/10)
]
optimizers_and_layers = [(optimizers[0], model.layers[:configs.unfreeze_index]),
                         (optimizers[1], model.layers[configs.unfreeze_index:])]

optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)


model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer=optimizer, metrics=["accuracy"])

for layer in model.layers:
    if layer.name in configs.unfreeze_layer_names and '_bn' not in layer.name:
        layer.trainable = True

print('Start Finetune the model, Finetune at : {}'.format(config.unfreeze))
history = model.fit(X_train,
                    validation_data=X_val,
                    epochs=finetune_epochs,
                    # callbacks=[reduce_lr],
                    # callbacks=[early_callback, wandb_callback],
                    # callbacks=[reduce_lr, wandb_callback],
                    callbacks=[wandb_callback],

                    )

print('Model trained in {:.1f}min'.format((time.time() - t0) / 60))
print('now is:  ' + now)
# model.save(now + '.h5')

# model_path = '/root/autodl-tmp/dl_project2/saved_model_' + now
# if not os.path.exists(model_path):
#     os.mkdir(model_path)
# model.save(model_path + '.h5')



