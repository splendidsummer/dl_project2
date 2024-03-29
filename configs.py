import tensorflow as tf
import random

import efficientnet_config
import resnet_config

"""
Setting basic configuration for this project
mainly including:
    1.
"""

BASE_DIR = './data/'
RAW_DATA_DIR = './data/indoorCVPR_09/Images/'
PROJECT_NAME = 'Feature_Extraciton_Resnet'
TEAM_NAME = 'unicorn_upc_dl'
TRAIN_FOLDER = './data/train/'
VAL_FOLDER = './data/val'
TEST_FOLDER = './data/test/'
seed = 10000
img_height, img_width, n_channels = 320, 320, 3

augment_config = {
    'augmentation': True,
    'random_ratio': 0.2,
    'img_resize': 256,
    # 'img_crop': 0.2,
    'img_crop_size': 256,
    'img_factor': 0.2,
    'convert_gray': True
}

input_shape = (img_height, img_width, n_channels)


class ValueSettingError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)
        self.errorinfo = ErrorInfo

    def __str__(self):  # using str method to print error type
        return self.errorinfo


value_error = ValueSettingError('Setting architecture or freeze stage wrong!!!')

data_classes = ['Oil on canvas', 'Graphite', 'Glass', 'Limestone', 'Bronze',
                'Ceramic', 'Polychromed wood', 'Faience', 'Wood', 'Gold', 'Marble',
                'Ivory', 'Silver', 'Etching', 'Iron', 'Engraving', 'Steel',
                'Woodblock', 'Silk and metal thread', 'Lithograph',
                'Woven fabric ', 'Porcelain', 'Pen and brown ink', 'Woodcut',
                'Wood engraving', 'Hand-colored engraving', 'Clay',
                'Hand-colored etching', 'Albumen photograph']

num_classes = len(data_classes)

wandb_config = {
    # "project_name": "CRB",
    "architecture": 'resnet50',
    'unfreeze': 'None',  # last_block
    # "epochs": 20,
    "freeze_epochs": 15,
    'finetune_ratio': 5,
    # "finetune_epochs": int(wandb.finetune_ratio * wandb.freeze_epochs),
    "batch_size": 32,
    'weight_decay': 0,  # 0.001, 0.0001
    'drop_rate': 0.2,
    # "learning_rate": [0.00001, 0.0001, 0.001, 0.01, 0.1],
    "learning_rate": 0.001,
    "epsilon": 1e-7,
    "amsgrad": False,
    "momentum": 0.0,   # how to set this together with lr???
    "nesterov": False,
    "activation": 'relu',  # 'selu', 'leaky_relu'(small leak:0.01, large leak:0.2), 'gelu',
    "initialization": "he_normal",
    "optimizer": 'adam',
    # "dropout": random.uniform(0.01, 0.80),
    # "normalization": True,
    # "early_stopping": True,
    # "augment": False,
    "num_hidden": 512,
    "augmentation": None,  # "random_crop", "random_ratation"
    "lr_scheduler": "None",
    }

# wandb_config.update(augment_config)
#
sweep_configuration = {
    'method': 'grid',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize',
        'name': 'validation_loss'},
    'parameters': {
        'batch_size': {'values': [32, 64, 128, 256]},
        'epochs': {'values': [20, 35, 50]},
        'weight_decay': {'values': [0, 0.01, 0.001, 0.0001, 0.00005]},
        'learning_rate': {'values': [0.01, 0.001, 0.0001, 0.00001]},
        'activation': {'values': ['relu', 'elu', 'selu', 'gelu']},
        'initialization': {'values': ['he_normal', 'glorot_uniform']}
     }
}

if wandb_config['architecture'] == 'resnet50' and wandb_config['unfreeze'] == 'last_stage':
    unfreeze_layer_names = resnet_config.last_stage_layers
    unfreeze_index = resnet_config.last_stage_index
elif wandb_config['architecture'] == 'resnet50' and wandb_config['unfreeze'] == 'last_2stage':
    unfreeze_layer_names = resnet_config.last_2stage_layers
    unfreeze_index = resnet_config.last_2stage_index
elif wandb_config['architecture'] == 'resnet50' and wandb_config['unfreeze'] == 'last_3stage':
    unfreeze_layer_names = resnet_config.last_3stage_layers
    unfreeze_index = resnet_config.last_3stage_index
elif wandb_config['architecture'] == 'resnet50' and wandb_config['unfreeze'] == 'last_4stage':
    unfreeze_layer_names = resnet_config.last_4stage_layers
    unfreeze_index = resnet_config.last_4stage_index
elif wandb_config['architecture'] == 'resnet50' and wandb_config['unfreeze'] == 'last_block':
    unfreeze_layer_names = resnet_config.last_block_layers
    unfreeze_index = resnet_config.last_block_index
elif wandb_config['architecture'] == 'resnet50' and wandb_config['unfreeze'] == 'None':
    unfreeze_index = resnet_config.last_index

elif wandb_config['architecture'] == 'efficientnetb1' and wandb_config['unfreeze'] == 'first_stage':
    unfreeze_index = efficientnet_config.first_stage_index

elif wandb_config['architecture'] == 'efficientnetb1' and wandb_config['unfreeze'] == 'second_stage':
    unfreeze_index = efficientnet_config.second_stage_index

elif wandb_config['architecture'] == 'efficientnetb1' and wandb_config['unfreeze'] == 'third_stage':
    unfreeze_index = efficientnet_config.third_stage_index

elif wandb_config['architecture'] == 'efficientnetb1' and wandb_config['unfreeze'] == 'fouth_stage':
    unfreeze_index = efficientnet_config.fouth_stage_index

elif wandb_config['architecture'] == 'efficientnetb1' and wandb_config['unfreeze'] == 'fifth_stage':
    unfreeze_index = efficientnet_config.fifth_stage_index

elif wandb_config['architecture'] == 'efficientnetb1' and wandb_config['unfreeze'] == 'sixth_stage':
    unfreeze_index = efficientnet_config.sixth_stage_index

elif wandb_config['architecture'] == 'efficientnetb1' and wandb_config['unfreeze'] == 'last_stage':
    unfreeze_index = efficientnet_config.last_stage_index
#
# else:
#     raise value_error
# #
# else:
#     unfreeze_layer_names = efficientnet_config.last_block_layers
#     unfreeze_index = efficientnet_config.last_block_index


