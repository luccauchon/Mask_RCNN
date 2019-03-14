import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
MODEL_DIR = os.path.join(ROOT_DIR, "logsV3")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


class AmateurConfig(Config):
    # Give the configuration a recognizable name
    NAME = "hq"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # IMAGE_MIN_DIM = 256
    # IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 128#32

    # Use a small epoch since the data is simple
    # STEPS_PER_EPOCH = 200

    # use small validation steps since the epoch is small
    # VALIDATION_STEPS = 30

    MASK_SHAPE = [28, 28]

    BASE_DIR = 'G:/AMATEUR/segmentation/22FEV2019/GEN_segmentation/'


config = AmateurConfig()
config.display()

os.environ['basedir_a']='C:/Temp2/'
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
seed = 10
test_size = 0.1
image_ids = [f for f in listdir(config.BASE_DIR) if isfile(join(config.BASE_DIR, f))]
train_list, val_list = train_test_split(image_ids, test_size=test_size, random_state=seed)
print ('train size='+str(len(train_list)))
print ('val size='+str(len(val_list)))


import luccauchon.data.dataset_util as dataset_util

# Training dataset
dataset_train = dataset_util.AmateurDatasetOnDiskMRCNN()
dataset_train.load(config.BASE_DIR, train_list,n=10)
dataset_train.prepare()

# Validation dataset
dataset_val = dataset_util.AmateurDatasetOnDiskMRCNN()
dataset_val.load(config.BASE_DIR, val_list,n=10)
dataset_val.prepare()

print (str(len(dataset_train.image_ids))+ ' images for training.')
print (str(len(dataset_val.image_ids))+ ' images for validating.')

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=6,
            layers='heads')