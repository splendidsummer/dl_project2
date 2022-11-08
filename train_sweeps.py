import tensorflow as tf
from utils import *
from cnn_models import *
import configs
from models import *
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os, sys, time, tqdm, datetime
from keras.optimizers import SGD, Adam
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping
import wandb
from wandb.keras import WandbCallback

sweep_id = wandb.sweep(sweep=configs.sweep_configuration, project="best_model_sweep_batchsize")


def onerun():
    wandb.init(
        project='cnn4_crb_bottleneck_sweep',
    )
    config = wandb.config
    lr = config.learning_rate
    epochs = config.epochs
    optimizer = config.optimizer
    batch_size = config.batch_size
    wd = config.weight_decay

    model = cnn4_crb_bottleneck(config)

    X_train = tf.keras.utils.image_dataset_from_directory(configs.train_folder,
    batch_size=batch_size,
    shuffle=True,
    seed=configs.seed)

    X_val = tf.keras.utils.image_dataset_from_directory(configs.val_folder,
    batch_size=batch_size,  # batch_size
    # image_size=(img_height, img_width), # resize
    shuffle=False,
    seed=configs.seed,)

    if wd == 0:
        optimizer = Adam(lr=lr)
    else:
        optimizer = tfa.optimizers.AdamW(
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            # epsilon=config.epsilon,
            # amsgrad=config.amsgrad,
        )

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=optimizer, metrics=["accuracy"])
    # Use early stopping
    reduce_lr = ReduceLROnPlateau(min_lr=0.00001)
    early_callback = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=1, mode='auto',
                      restore_best_weights=True)

    wandb_callback = WandbCallback(save_model=True)

    print('Starting training!!')

    history = model.fit(X_train,
                        validation_data=X_val,
                        epochs=epochs,
                        callbacks=[reduce_lr, wandb_callback],
                        )


wandb.agent(sweep_id=sweep_id, function=onerun)

