import sys
import os

if os.name is not 'nt':
    assert os.environ["CUDA_VISIBLE_DEVICES"] is not None
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    return tf.Session(config=config)


import keras

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

import time
from multiprocessing import freeze_support
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# https://stackoverflow.com/questions/17053671/python-how-do-you-stop-numpy-from-multithreading
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np

from loguru import logger
import mlflow

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
assert os.path.exists(COCO_MODEL_PATH), str(COCO_MODEL_PATH)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize
from pathlib import Path
import cv2 as cv2

logger.debug('Numpy: ' + str(np.__version__))
logger.debug('Tensorflow: ' + str(tf.__version__))
logger.debug('keras.__version__=' + str(keras.__version__))
logger.debug('Using GPU ' + str(os.environ["CUDA_VISIBLE_DEVICES"]) + '  Good luck...')
logger.debug('Using conda env: ' + str(Path(sys.executable).as_posix().split('/')[-3]) + ' [' + str(Path(sys.executable).as_posix()) + ']')


############################################################
#  Configurations
############################################################


class ApprantiConfig(Config):
    NAME = "coco"
    IMAGES_PER_GPU = 4
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 1
    LOSS_WEIGHTS = {"rpn_class_loss": 1., "rpn_bbox_loss": 1., "mrcnn_class_loss": 1., "mrcnn_bbox_loss": 1., "mrcnn_mask_loss": 1.}
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640
    USE_MINI_MASK = True


class ApprantiInferenceConfig(ApprantiConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0


############################################################
#  Dataset
############################################################


class ApprantiDataset(utils.Dataset):
    def load_appranti(self, dataset_dir, subset, year, config, class_ids=None, return_coco=False, limit_number_images=None):
        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)
        assert os.path.exists(image_dir)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            assert limit_number_images is None
            image_ids = list(coco.imgs.keys())

        if limit_number_images is not None:
            image_ids = image_ids[0:limit_number_images]

        assert len(class_ids) + 1 == config.NUM_CLASSES, str(len(class_ids) + 1) + ' != ' + str(config.NUM_CLASSES)
        logger.debug(str(len(image_ids)) + ' images in ' + subset + ' dataset splitted in ' + str(len(class_ids)) + ' classes.')
        if limit_number_images is not None:
            assert len(image_ids) == limit_number_images

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None)))

        if return_coco:
            return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            assert False
            return super(ApprantiDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                    assert False
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            assert False
            return super(ApprantiDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            assert False
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            assert False
            super(ApprantiDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################
def evaluate(args_):
    None


############################################################
#  Training
############################################################
def train(args_, model, config, is_mlflow=False, run_name=None, overfit=False):
    dataset_train = ApprantiDataset()
    dataset_train.load_appranti(dataset_dir=args_.dataset, subset="train", year=args_.year, class_ids=None, config=config, limit_number_images=(1 if overfit else None))
    dataset_train.prepare()

    dataset_val = ApprantiDataset()
    dataset_val.load_appranti(dataset_dir=args_.dataset, subset="val", year=args_.year, class_ids=None, config=config)
    dataset_val.prepare()

    if args_.eyes_of_the_net is not None:
        for dataset, name in zip((dataset_train, dataset_val), ('train', 'val')):
            output_dir = args_.eyes_of_the_net + '/' + name + '/'
            os.makedirs(output_dir, exist_ok=True)
            for img_info in dataset.image_info:
                source_id = str(img_info['id'])
                image_id = dataset.image_from_source_map["coco.{}".format(source_id)]
                image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
                an_image = visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names, show_bbox=False, only_return_image=True)
                cv2.imwrite(output_dir + str(image_id) + '.jpg', an_image)

    config.STEPS_PER_EPOCH = 500
    assert config.STEPS_PER_EPOCH * config.IMAGES_PER_GPU >= len(dataset_train.image_ids)
    config.VALIDATION_STEPS = 100
    assert config.VALIDATION_STEPS * config.IMAGES_PER_GPU >= len(dataset_val.image_ids)

    custom_callbacks = [
        # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=7, verbose=1, mode='auto', min_delta=0.01, cooldown=0, min_lr=10e-7),

        # ['val_loss', 'val_rpn_class_loss', 'val_rpn_bbox_loss', 'val_mrcnn_class_loss', 'val_mrcnn_bbox_loss', 'val_mrcnn_mask_loss', 'loss', 'rpn_class_loss', 'rpn_bbox_loss',
        #     'mrcnn_class_loss', 'mrcnn_bbox_loss', 'mrcnn_mask_loss'])
        # keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: logger.debug(logs.keys())),
    ]

    if is_mlflow:
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(
            {'model_dir': str(model.log_dir), 'n_train': str(len(dataset_train.image_ids)), 'n_val': str(len(dataset_val.image_ids)), 'os': os.name, 'dataset': 'appranti',
             'CUDA_VISIBLE_DEVICES': os.environ["CUDA_VISIBLE_DEVICES"], 'class_names': str(dataset_train.class_names), 'init_with_model': str(args_.model)})
        mlflow.log_params(config.get_parameters())

        def my_log_func(epoch, logs):
            logger.debug(logs)
            mlflow.log_metrics(logs)
            mlflow.log_metrics({'epoch': float(epoch)})

        custom_callbacks.append(keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: my_log_func(epoch, logs)))

    logger.debug("Training network heads")
    epoch = 40
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epoch, layers='heads', augmentation=None, custom_callbacks=custom_callbacks,
                model_check_point=(False if overfit else True))

    logger.debug("Fine tune Resnet stage 4 and up")
    epoch = 120
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epoch, layers='4+', augmentation=None, custom_callbacks=custom_callbacks,
                model_check_point=(False if overfit else True))

    logger.debug("Fine tune all layers")
    epoch = 160 + 250
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=epoch, layers='all', augmentation=None, custom_callbacks=custom_callbacks,
                model_check_point=(False if overfit else True))


############################################################
#
############################################################
def start(args_):
    logger.debug("Command: " + str(args_.command))
    logger.debug("Model: " + str(args_.model))
    logger.debug("Dataset: " + str(args_.dataset))
    logger.debug("Year: " + str(args_.year))
    logger.debug("Logs and training weights: " + str(args_.logs))

    try:
        run_name = args_.run_name
        if os.name is not 'nt':
            is_mlflow = True
            mlflow.set_tracking_uri("http://10.7.248.206:3389")
            mlflow.set_experiment("appranti_maskrcnn")
            logger.info('Enabling MLFLOW. Run name:' + str(run_name))
        else:
            is_mlflow = False
            logger.info('Disabling MLFLOW on my computer.:)')
    except Exception as e:
        logger.warning('Disabling MLFLOW.')
        is_mlflow = False

    # Configurations
    if args_.command == "train":
        config = ApprantiConfig()
    else:
        config = ApprantiInferenceConfig()

    # logger.debug(config.display_in_str())

    # Create model
    if 'train' in args_.command:
        my_model = modellib.MaskRCNN(mode="training", config=config, model_dir=args_.logs)
    elif 'inference' in args_.command:
        my_model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args_.logs)
    else:
        assert False

    # Select weights file to load
    exclude = []
    if args_.model.lower() == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model_path = COCO_MODEL_PATH
        exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
    elif args_.model.lower() == "last":
        # Find last trained weights
        model_path = my_model.find_last()
    elif args_.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = my_model.get_imagenet_weights()
    else:
        model_path = None

    # Load weights
    if model_path is not None:
        logger.debug("Loading weights from " + str(model_path))
        my_model.load_weights(model_path, by_name=True, exclude=exclude)

    # Train or evaluate
    if 'train' in args_.command:
        train(args_=args_, model=my_model, config=config, is_mlflow=is_mlflow, run_name=run_name, overfit=(True if 'overfit' in args_.command else False))
    elif args_.command == "evaluate":
        evaluate(args_=args_)
    else:
        assert False


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on Appranti Dataset.')
    parser.add_argument("command", metavar="<command>", help="'train','train_overfit','evaluate'")
    parser.add_argument('--dataset', required=True, metavar="/path/to/coco/", help='Directory of the Appranti dataset')
    parser.add_argument('--year', required=False, default=2019, metavar="<year>", help='Year of the Appranti dataset (2019 or ...) (default=2019)')
    parser.add_argument('--model', required=True, metavar="/path/to/weights.h5", help="'coco', 'last' or 'imagenet'")
    parser.add_argument('--logs', required=False, default=os.path.join(ROOT_DIR, "logs_appranti"), metavar="/path/to/logs/", help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--eyes_of_the_net', required=False, help='Vizualization of tensors given to the net')
    parser.add_argument('--run_name', required=True, metavar="rn_MaskRCNN", help='the run name for mlflow')

    args = parser.parse_args()

    freeze_support()
    start(args)
