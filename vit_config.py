ViT_B16_layer_names = ['patch_embed', 'cls_pos', 'dropout', 'encoderblock_0',
                       'encoderblock_1', 'encoderblock_2', 'encoderblock_3',
                       'encoderblock_4', 'encoderblock_5', 'encoderblock_6',
                       'encoderblock_7', 'encoderblock_8', 'encoderblock_9',
                       'encoderblock_10', 'encoderblock_11', 'encoder_norm',
                       'activation_24', 'head']

ViT_L16_layer_names = ['patch_embed', 'cls_pos', 'dropout', 'encoderblock_0',
                       'encoderblock_1', 'encoderblock_2', 'encoderblock_3',
                       'encoderblock_4', 'encoderblock_5', 'encoderblock_6',
                       'encoderblock_7', 'encoderblock_8', 'encoderblock_9',
                       'encoderblock_10', 'encoderblock_11', 'encoderblock_12',
                       'encoderblock_13', 'encoderblock_14', 'encoderblock_15',
                       'encoderblock_16', 'encoderblock_17', 'encoderblock_18',
                       'encoderblock_19', 'encoderblock_20', 'encoderblock_21',
                       'encoderblock_22', 'encoderblock_23', 'encoder_norm',
                       'activation_24', 'head']


freeze_layers = ['encoderblock_21', 'encoderblock_22', 'encoderblock_23', 'encoder_norm',]