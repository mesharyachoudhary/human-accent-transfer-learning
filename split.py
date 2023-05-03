import os
import shutil
import random
import math

# Set the path to your main directory
main_dir = './clips/'

# Set the percentage split for train and validation sets
train_pct = 0.9
val_pct = 1 - train_pct

# Get a list of all the subdirectories in the main directory
classes = [f.name for f in os.scandir(main_dir) if f.is_dir()]

# Create 'train' and 'val' directories if they don't exist
train_dir = os.path.join(main_dir, 'train')
val_dir = os.path.join(main_dir, 'validation')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(val_dir):
    os.mkdir(val_dir)

# Loop through each subdirectory and split the files into train and validation sets
for cls in classes:
    cls_path = os.path.join(main_dir, cls)
    train_cls_path = os.path.join(train_dir, cls)
    val_cls_path = os.path.join(val_dir, cls)
    if not os.path.exists(train_cls_path):
        os.mkdir(train_cls_path)
    if not os.path.exists(val_cls_path):
        os.mkdir(val_cls_path)
    files = os.listdir(cls_path)
    random.shuffle(files)
    x = math.floor(train_pct*len(files))
    print(cls, x)
    train_files = files[:x]
    print(len(train_files))
    val_files = files[x:]
    for f in train_files:
        src = os.path.join(cls_path, f)
        dst = os.path.join(train_cls_path, f)
        shutil.move(src, dst)
    for f in val_files:
        src = os.path.join(cls_path, f)
        dst = os.path.join(val_cls_path, f)
        shutil.move(src, dst)
    shutil.rmtree(cls_path)
