import tensorflow_hub as hub
from models import *
import os, sys, time, tqdm, datetime
import math
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

# Load model from TFHub into KerasLayer
model_url = "https://tfhub.dev/google/bit/m-r50x1/1"
module = hub.load('/root/autodl-tmp/dl_project2/bit/bit_m-r50x1_1/')


class MyBiTModel(tf.keras.Model):
  """BiT with a new head."""

  def __init__(self, num_classes, module):
    super().__init__()

    self.num_classes = num_classes
    self.hidden = Dense(512, activation='relu')
    self.dropout = Dropout(0.2)
    self.head = tf.keras.layers.Dense(num_classes, kernel_initializer='zeros')
    self.bit_model = module
    self._freeze_backbone()

  def call(self, images):
    # No need to cut head off since we are using feature extractor model
    bit_embedding = self.bit_model(images)
    out = self.dropout(self.hidden(bit_embedding))
    return self.head(out)

  def _freeze_backbone(self):
      self.bit_model.trainable = False
      print(type(self.bit_model))
      print('model trainable: ', self.bit_model.trainable)


model = MyBiTModel(num_classes=29, module=module)
set_seed(configs.seed)

now = datetime.datetime.now()
now = now.strftime('%Y%m%d%H%M%S')
is_gpu = tf.config.list_physical_devices('GPU') is not None
wandb_dir = '/root/autodl-tmp/dl_project2/wandb_logs' if is_gpu else \
    'D:/UPC_Course3/DL/dl_project1/wandb_logs'

# initialize wandb logging to your project
wandb.init(
    job_type='Trying different architecture',
    project=configs.PROJECT_NAME,
    dir=wandb_dir,
    entity=configs.TEAM_NAME,
    config=configs.wandb_config,
    # sync_tensorboard=True,
    name='BIT' + now,
    notes='min_lr=0.00001',
    ####
)

config = wandb.config
batch_size = config.batch_size
freeze_epochs = config.freeze_epochs
# finetune_epochs = config.finetune_epochs
lr = config.learning_rate
weight_decay = config.weight_decay
# early_stopping = config.early_stopping
activation = config.activation
# augment = config.augment

print('Build Training dataset')
X_train = tf.keras.utils.image_dataset_from_directory(
    configs.TRAIN_FOLDER,
    batch_size=config.batch_size,  # batch_size
    image_size=(configs.img_height, configs.img_width),  # resize
    shuffle=True,
    seed=configs.seed
)

print('Build Validation dataset')
X_val = tf.keras.utils.image_dataset_from_directory(
    configs.VAL_FOLDER,
    batch_size=config.batch_size,  # batch_size
    image_size=(configs.img_height, configs.img_width),  # resize
    shuffle=False,
    seed=configs.seed,
)

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

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=0.00001
)

# optimizer = tf.keras.optimizers.Adam(learning_rate=lr)   # 下一次测试的参数
optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])

print('Start Training Classifier!')
t0 = time.time()

history = model.fit(X_train,
                    validation_data=X_val,
                    epochs=30,
                    callbacks=[reduce_lr_callback, wandb_callback],
                    )

# print(model.trainable_variables)




