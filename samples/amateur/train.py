import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if os.name == 'nt':
    os.environ['basedir_a'] = 'F:/Temp2/'
    os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = 'C:\\Users\\cj3272\\PortableGit\\bin\\git.exe'
else:
    os.environ['basedir_a'] = '/gpfs/home/cj3272/tmp/'

import luccauchon.data.C as C
import PIL
import keras
import logging
import luccauchon.data.__MYENV__ as E  # Charge en mémoire des variables d'environnement.

E.APPLICATION_LOG_LEVEL = logging.DEBUG
if C.is_running_on_casir():
    E.APPLICATION_LOG_LEVEL = logging.INFO
log = E.setup_logger(logger_name=__name__, _level=E.APPLICATION_LOG_LEVEL)

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
import mlflow
import luccauchon.data.dataset_util as dataset_util
from samples.amateur.config import AmateurTrain
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
import mrcnn.model as model_lib


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def main():
    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    log.info('keras.__version__=' + str(keras.__version__))
    log.info('tf.__version__=' + str(tf.__version__))
    log.info('PIL.__version__=' + str(PIL.__version__))
    log.info('Using GPU ' + str(os.environ["CUDA_VISIBLE_DEVICES"]) + '  Good luck...')

    mlflow.set_tracking_uri("http://10.7.248.206:3389")
    mlflow.set_experiment("TEST")
    mlflow.start_run()

    model_dir = 'F:/AMATEUR/models_mask_rcnn/'
    coco_model_path = 'C:/Users/cj3272/PycharmProjects/Mask_RCNN/mask_rcnn_coco.h5'
    if C.is_running_on_casir():
        model_dir = '/gpfs/home/cj3272/amateur/modeles/segmentation/MaskRCNN_win/'
        coco_model_path = '/gpfs/home/cj3272/amateur/modeles/segmentation/MaskRCNN_win/mask_rcnn_coco.h5'

    config = AmateurTrain()
    assert config.BASE_DIR is None
    if C.is_running_on_casir():
        config.BASE_DIR = '/gpfs/groups/gc014a/AMATEUR/dataset/segmentation/30MAI2019/GEN_segmentation/'
    else:
        config.BASE_DIR = 'F:/AMATEUR/segmentation/13JUIN2019/GEN_segmentation/'

    # Datasets
    seed = 10
    val_size = 0.2
    image_ids = [f for f in listdir(config.BASE_DIR) if isfile(join(config.BASE_DIR, f)) and f.endswith('gz')]
    train_list, val_list = train_test_split(image_ids, test_size=val_size, random_state=seed)
    log.debug('train size=' + str(len(train_list)))
    log.debug('val size=' + str(len(val_list)))

    # Training dataset
    dataset_train = dataset_util.AmateurDatasetOnDiskMRCNN()
    dataset_train.load(config.BASE_DIR, train_list)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = dataset_util.AmateurDatasetOnDiskMRCNN()
    dataset_val.load(config.BASE_DIR, val_list)
    dataset_val.prepare()

    log.info(str(len(dataset_train.image_ids)) + ' images for training.')
    log.info(str(len(dataset_val.image_ids)) + ' images for validating.')

    # Create model in training mode
    model = model_lib.MaskRCNN(mode="training", config=config, model_dir=model_dir)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(coco_model_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    # model.train(dataset_train, dataset_val,                learning_rate=config.LEARNING_RATE,                epochs=6,                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    custom_callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1,mode='auto', min_delta=0.01, cooldown=0, min_lr=10e-7)]
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=150, layers="all", custom_callbacks=custom_callbacks)


if __name__ == "__main__":
    main()
