import tensorflow as tf
import os, sys, time, tqdm, datetime
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
    configs.TEST_FOLDER,
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
model = build_resnet50()
print('Trainable variables in model: ', len(model.trainable_variables))

optimizer1 = Adam(  # do we need to change the epsilon??
        lr=config.learning_rate,
        epsilon=config.epsilon,
        amsgrad=config.amsgrad
    )

optimizer2 = Adam(
    lr=config.learning_rate * 0.1,
    epsilon=config.epsilon,
    amsgrad=config.amsgrad
)

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

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)
reduce_lr = ReduceLROnPlateau(min_lr=0.00001)

# Firstly finetune the classifer with fixing the weights of base_model
# ???? specify the index of .....
model.layers[4].trainable = False
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer=optimizer1, metrics=["accuracy"])

print('Start Training!')
t0 = time.time()
print('training model')

history = model.fit(X_train,
                    validation_data=X_val,
                    epochs=config.epochs,
                    # callbacks=[reduce_lr],
                    # callbacks=[early_callback, wandb_callback],
                    callbacks=[reduce_lr, wandb_callback],
                    )

optimizers = [
    tf.keras.optimizers.Adam(learning_rate=lr),
    tf.keras.optimizers.Adam(learning_rate=lr/10)
]
optimizers_and_layers = [(optimizers[0], model.layers[0]), (optimizers[1], model.layers[1:])]
optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer=optimizer1, metrics=["accuracy"])

history = model.fit(X_train,
                    validation_data=X_val,
                    epochs=config.epochs,
                    # callbacks=[reduce_lr],
                    # callbacks=[early_callback, wandb_callback],
                    callbacks=[reduce_lr, wandb_callback],
                    )

print('Model trained in {:.1f}min'.format((time.time() - t0) / 60))
print('now is:  ' + now)
# model.save(now + '.h5')

model_path = '/root/autodl-tmp/dl_project2/saved_model_' + now
if not os.path.exists(model_path):
    os.mkdir(model_path)
model.save(model_path + '.h5')
