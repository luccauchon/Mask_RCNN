import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# set the modified tf session as backend in keras
import keras

keras.backend.tensorflow_backend.set_session(get_session())

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
MODEL_DIR = 'F:/AMATEUR/models_mask_rcnn/hq20190528T1316/'

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import numpy as np
import scipy as scipy

os.environ['basedir_a'] = 'F:/Temp2/'
import PIL as PIL
import luccauchon.data.__MYENV__ as E
import logging

E.APPLICATION_LOG_LEVEL = logging.DEBUG
log = E.setup_logger(logger_name=__name__, _level=E.APPLICATION_LOG_LEVEL)

try:
    import cv2 as cv2
except:
    log.error('Need to install opencv.')


class AmateurConfig(Config):
    # Give the configuration a recognizable name
    NAME = "hq"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

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

    BASE_DIR = 'F:/AMATEUR/segmentation/22FEV2019/GEN_segmentation/'


class InferenceConfig(AmateurConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
my_model = modellib.MaskRCNN(mode="inference",
                             config=inference_config,
                             model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = MODEL_DIR + '/mask_rcnn_hq_0018.h5'

# Load trained weights
log.debug("Loading weights from " + model_path)
my_model.load_weights(model_path, by_name=True)
my_model.keras_model.summary()
import luccauchon.data.dataset_util as dataset_util
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split

seed = 12
test_size = 0.025
image_ids = [f for f in listdir(inference_config.BASE_DIR) if isfile(join(inference_config.BASE_DIR, f))]
_, test_list = train_test_split(image_ids, test_size=test_size, random_state=seed)
log.debug('test size=' + str(len(test_list)))

# Test dataset
dataset_test = dataset_util.AmateurDatasetMemoryMRCNN()
dataset_test.load(inference_config.BASE_DIR, test_list)
dataset_test.prepare()

log.debug('keras.__version__=' + str(keras.__version__))
log.debug('np.__version__=' + str(np.__version__))
log.debug('scipy.__version__=' + str(scipy.__version__))
log.debug('tf.__version__=' + str(tf.__version__))
log.debug('PIL.__version__=' + str(PIL.__version__))
log.debug('Using GPU ' + str(os.environ["CUDA_VISIBLE_DEVICES"]) + '  Good luck...')

log.debug(str(len(dataset_test.image_ids)) + ' images for testing.')

base_dir_images = 'F:/AMATEUR/fichiers_pour_tests/'
result_dir_images = 'F:/AMATEUR/results/'
list_of_images = [f for f in os.listdir(base_dir_images)]
for image_file in list_of_images:
    im = cv2.imread(base_dir_images + image_file)
    # im = PIL.Image.open(base_dir_images + image_file)
    # im.show()
    ima = im  # np.asarray(im)
    results = my_model.detect([ima], verbose=0)
    assert 1 == len(results)
    r = results[0]
    my_result = visualize.display_instances(ima, r['rois'], r['masks'], r['class_ids'], dataset_test.class_names, r['scores'], only_return_image=True)
    assert isinstance(my_result, np.ndarray)
    # my_image = PIL.Image.fromarray(my_result)
    cv2.imwrite(result_dir_images+image_file ,my_result)
