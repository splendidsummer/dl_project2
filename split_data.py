"""
This file is used to split the raw image data in the original folder
("indoorCVPR_09") to corresponding "./data/train/" & "./data/test/" directory
according to "TestImages.txt" and "TrainImage.txt"
"""
import configs, os, shutil
import pandas as pd

classes = []
for path in os.listdir(configs.RAW_DATA_DIR):
    classes.append(path)

train_txt = configs.BASE_DIR + 'TrainImages.txt'
test_txt = configs.BASE_DIR + 'TestImages.txt'

org_train_paths = []
train_classes =[]
org_test_paths = []
test_classes = []


with open(train_txt) as f:
    lines = f.readlines()
    for line in lines:
        org_train_paths.append(line.strip())
        train_classes.append(line.split('/')[0])
print(len(org_train_paths))

with open(test_txt) as f:
    lines = f.readlines()
    for line in lines:
        org_test_paths.append(line.strip())
        test_classes.append(line.split('/')[0])
print(len(org_test_paths))

# full_train_paths = [configs.RAW_DATA_DIR + path for path in org_train_paths]
# full_test_paths = [configs.RAW_DATA_DIR + path for path in org_test_paths]

train_cls_dirs = [configs.TRAIN_FOLDER + i + '/' for i in classes]
test_cls_dirs = [configs.TEST_FOLDER + i + '/' for i in classes]

for directory in train_cls_dirs:
    if not os.path.exists(directory):
        os.mkdir(directory)

for directory in test_cls_dirs:
    if not os.path.exists(directory):
        os.mkdir(directory)

train_df = pd.read_csv('train_stats.csv')
test_df = pd.read_csv('test_stats.csv')

train_df['organized_path'] = train_df['path'].map(lambda x: '/'.join(x.split('/')[-2:]))
train_df.organized_path = train_df['organized_path'].map(lambda x:  configs.TRAIN_FOLDER + x)

test_df['organized_path'] = test_df['path'].map(lambda x: '/'.join(x.split('/')[-2:]))
test_df.organized_path = test_df['organized_path'].map(lambda x: configs.TEST_FOLDER + x)

for org_path, path in zip(train_df.path, train_df.organized_path):
    shutil.copyfile(org_path, path)

for org_path, path in zip(test_df.path, test_df.organized_path):
    shutil.copyfile(org_path, path)

