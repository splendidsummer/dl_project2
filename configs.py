import tensorflow as tf
import random

"""
Setting basic configuration for this project
mainly including:
    1.
"""

PROJECT_NAME = 'Report_trails'
TEAM_NAME = 'unicorn_upc_dl'
train_folder = './data/train'
val_folder = './data/test'
test_folder = './data/val'
seed = 168
img_height, img_width, n_channels = 256, 256, 3

augment_config = {
    'augmentation': False,
    'random_ratio': 0.2,
    'img_resize': 200,
    'img_crop': 0.2,
    'img_factor': 0.2,
    'convert_gray': True
}

input_shape = (img_height, img_width, n_channels)

data_columns = ['Image file', 'Medium', 'Museum', 'Museum-based instance ID', 'Subset',
                'Width', 'Height', 'Product size', 'Aspect ratio']

data_classes = ['Oil on canvas', 'Graphite', 'Glass', 'Limestone', 'Bronze',
                'Ceramic', 'Polychromed wood', 'Faience', 'Wood', 'Gold', 'Marble',
                'Ivory', 'Silver', 'Etching', 'Iron', 'Engraving', 'Steel',
                'Woodblock', 'Silk and metal thread', 'Lithograph',
                'Woven fabric ', 'Porcelain', 'Pen and brown ink', 'Woodcut',
                'Wood engraving', 'Hand-colored engraving', 'Clay',
                'Hand-colored etching', 'Albumen photograph']

# initializer = tf.keras.initializers.LecunNormal()

wandb_config = {
    # "project_name": "CRB",
    "architecture": 'CONV',
    "epochs": 50,
    "batch_size": 128,
    'weight_decay': 0,
    'drop_rate': 0.2,
    # "learning_rate": [0.00001, 0.0001, 0.001, 0.01, 0.1],
    "learning_rate": 0.0001,
    "epsilon": 1e-7,
    "amsgrad": False,
    "momentum": 0.0,   # how to set this together with lr???
    "nesterov": False,
    "activation": 'selu',  # 'selu', 'leaky_relu'(small leak:0.01, large leak:0.2), 'gelu',
    "initialization": "he_normal",
    "optimizer": 'adam',
    # "dropout": random.uniform(0.01, 0.80),
    "normalization": True,
    "early_stopping": True,
    "augment": False
    }

wandb_config.update(augment_config)
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

sweep_configuration = {
    'method': 'grid',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize',
        'name': 'val_acc'},
    'parameters': {
        'batch_size': {'values': [32, 64, 128, 256]},
        'epochs': {'values': [35]},
        'weight_decay': {'values': [0.0001]},
        'learning_rate': {'values': [0.001]},
        'activation': {'values': ['gelu']},
        'initialization': {'values': ['he_normal']},
        'optimizer': {'values': ['adam']}
     }
}



