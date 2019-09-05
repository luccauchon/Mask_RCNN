import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if os.name == 'nt':
    os.environ['basedir_a'] = 'F:/Temp2/'
    os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = 'C:\\Users\\cj3272\\PortableGit\\bin\\git.exe'
else:
    os.environ['basedir_a'] = '/gpfs/home/cj3272/tmp/'
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# https://stackoverflow.com/questions/17053671/python-how-do-you-stop-numpy-from-multithreading
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import luccauchon.data.C as C
import PIL
import keras

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
import mlflow as mlflow
import luccauchon.data.dataset_util as dataset_util
from samples.amateur.config import AmateurTrain
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
import mrcnn.model as model_lib


def main():
    print('keras.__version__=' + str(keras.__version__))
    print('tf.__version__=' + str(tf.__version__))
    print('PIL.__version__=' + str(PIL.__version__))
    print('Using GPU ' + str(os.environ["CUDA_VISIBLE_DEVICES"]) + '  Good luck...')

    model_dir = 'F:/AMATEUR/models_mask_rcnn/'
    coco_model_path = 'C:/Users/cj3272/PycharmProjects/Mask_RCNN/mask_rcnn_coco.h5'
    if C.is_running_on_casir():
        model_dir = '/gpfs/home/cj3272/amateur/modeles/segmentation/MaskRCNN_win/logs/'
        coco_model_path = '/gpfs/home/cj3272/amateur/modeles/segmentation/MaskRCNN_win/mask_rcnn_coco.h5'

    config = AmateurTrain()
    assert config.BASE_DIR is None
    dataset_id = '30MAI2019'
    run_name = 'training'
    if C.is_running_on_casir():
        config.BASE_DIR = '/gpfs/groups/gc014a/AMATEUR/dataset/segmentation/' + dataset_id + '/GEN_segmentation/'
        config.IMAGES_PER_GPU = 4
        config.STEPS_PER_EPOCH = 1000
        config.VALIDATION_STEPS = 200
        n_train = None
        n_val = None
    else:
        config.BASE_DIR = 'F:/AMATEUR/segmentation/13JUIN2019/GEN_segmentation/'
        config.IMAGES_PER_GPU = 1
        config.IMAGE_MIN_DIM = 400
        config.IMAGE_MAX_DIM = 512
        n_train = 1
        n_val = 1

    # Datasets
    seed = 10
    val_size = 0.2
    image_ids = [f for f in listdir(config.BASE_DIR) if isfile(join(config.BASE_DIR, f)) and f.endswith('gz')]
    train_list, val_list = train_test_split(image_ids, test_size=val_size, random_state=seed)
    print('train size=' + str(len(train_list)))
    print('val size=' + str(len(val_list)))

    # Training dataset
    dataset_train = dataset_util.AmateurDatasetOnDiskMRCNN()
    # if C.is_running_on_casir():
    #    dataset_train = dataset_util.AmateurDatasetMemoryMRCNN()
    dataset_train.load(config.BASE_DIR, train_list, n=n_train)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = dataset_util.AmateurDatasetOnDiskMRCNN()
    dataset_val.load(config.BASE_DIR, val_list, n=n_val)
    dataset_val.prepare()

    print(str(len(dataset_train.image_ids)) + ' images for training.')
    print(str(len(dataset_val.image_ids)) + ' images for validating.')
    assert config.STEPS_PER_EPOCH * config.IMAGES_PER_GPU >= len(dataset_train.image_ids)
    assert config.VALIDATION_STEPS * config.IMAGES_PER_GPU >= len(dataset_val.image_ids), ''

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

    mlflow.set_tracking_uri("http://10.7.248.206:3389")
    mlflow.set_experiment("AMATEUR_SEGMENTATION")
    mlflow.start_run(run_name=run_name)
    mlflow.log_params({'model_dir': str(model_dir), 'n_train': str(n_train), 'n_val': str(n_val), 'init_with': init_with, 'os': os.name, 'dataset_id': dataset_id,
                       'CUDA_VISIBLE_DEVICES': os.environ["CUDA_VISIBLE_DEVICES"]})
    mlflow.log_params(config.get_parameters())

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    # model.train(dataset_train, dataset_val,                learning_rate=config.LEARNING_RATE,                epochs=6,                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    custom_callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=7, verbose=1, mode='auto', min_delta=0.01, cooldown=0, min_lr=10e-7),
                        keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: print(logs.keys())),
                        keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: mlflow.log_metrics(logs))]
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=150, layers="all", custom_callbacks=custom_callbacks)


if __name__ == "__main__":
    main()
