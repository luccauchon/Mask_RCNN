import os
if os.name == 'nt':
    os.environ['basedir_a'] = 'F:/Temp2/'
else:
    os.environ['basedir_a'] = '/gpfs/home/cj3272/tmp/'

import luccauchon.data.C as C
import PIL
import keras
from samples.amateur.config import AmateurInference
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
import luccauchon.data.dataset_util as dataset_util
import luccauchon.data.__MYENV__ as E  # Charge en mémoire des variables d'environnement.
import logging

E.APPLICATION_LOG_LEVEL = logging.DEBUG
if C.is_running_on_casir():
    E.APPLICATION_LOG_LEVEL = logging.INFO
log = E.setup_logger(logger_name=__name__, _level=E.APPLICATION_LOG_LEVEL)
import mrcnn.model as model_lib
from mrcnn import utils
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def compute_batch_ap(model, dataset, image_ids, config, verbose=1):
    assert isinstance(model, model_lib.MaskRCNN)
    APs = []
    buckets = [image_ids[i:i + config.BATCH_SIZE] for i in range(0, len(image_ids), config.BATCH_SIZE)]
    for images_id in buckets:
        if len(images_id) != config.BATCH_SIZE:
            continue
        images = []
        images_meta = []
        for image_id in images_id:
            # Load image
            log.debug('loading image %s' % image_id)
            image, image_meta, gt_class_id, gt_bbox, gt_mask = model_lib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
            images.append(image)
            images_meta.append(image_meta)

        # Run object detection
        results = model.detect_molded(np.stack(images, axis=0), np.stack(images_meta, axis=0), verbose=0)
        assert config.BATCH_SIZE == len(results)

        # Compute AP over range 0.5 to 0.95
        for r, image_id, image_meta in zip(results, images_id, images_meta):
            ap = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'], r['scores'], r['masks'], verbose=0)
            APs.append(ap)
            if verbose:
                info = dataset.image_info[image_id]
                meta = model_lib.parse_image_meta(image_meta[np.newaxis, ...])
                log.debug("{:3} {}   AP: {:.2f}".format(meta["image_id"][0], meta["original_image_shape"][0], ap))

    return APs


def main():
    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    log.debug('keras.__version__=' + str(keras.__version__))
    log.debug('tf.__version__=' + str(tf.__version__))
    log.debug('PIL.__version__=' + str(PIL.__version__))
    log.debug('Using GPU ' + str(os.environ["CUDA_VISIBLE_DEVICES"]) + '  Good luck...')

    if C.is_running_on_casir():
        # Root directory of the project
        root_dir = os.path.abspath("./Mask_RCNN")
        # Directory to save logs and trained model
        model_dir = '/gpfs/home/cj3272/amateur/modeles/segmentation/pfe-ireq-master/mask-rcnn/Mask_RCNN/logsV2/hq20190619T1127/'
    else:
        model_dir = 'F:/AMATEUR/models_mask_rcnn/hq20190619T1127/'

    config = AmateurInference()
    if C.is_running_on_casir():
        config.BASE_DIR = '/gpfs/groups/gc014a/AMATEUR/dataset/segmentation/30MAI2019/GEN_segmentation/'
        config.IMAGES_PER_GPU = 12
    else:
        config.BASE_DIR = 'F:/AMATEUR/segmentation/13JUIN2019/GEN_segmentation/'
        config.IMAGES_PER_GPU = 12

    seed = 11
    test_size = 0.99
    image_ids = [f for f in listdir(config.BASE_DIR) if isfile(join(config.BASE_DIR, f)) and f.endswith('gz')]
    _, test_list = train_test_split(image_ids, test_size=test_size, random_state=seed)
    log.info('found %s files for test in %s' % (str(len(test_list)), config.BASE_DIR))

    # Test dataset
    if C.is_running_on_casir():
        dataset_test = dataset_util.AmateurDatasetOnDiskMRCNN()
    else:
        dataset_test = dataset_util.AmateurDatasetOnDiskMRCNN()
    dataset_test.load(config.BASE_DIR, test_list, n=config.IMAGES_PER_GPU * 10)
    dataset_test.prepare()

    log.info(str(len(dataset_test.image_ids)) + ' images for testing.')

    # Recreate the model in inference mode
    model = model_lib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
    for epoch in range(1, 150, 1):
        model_path = model_dir + ('/mask_rcnn_hq_{:04d}.h5'.format(epoch))
        if not os.path.isfile(model_path):
            continue
        log.debug('Loading weights of ' + model_path + '...')
        model.load_weights(model_path, by_name=True)
        log.debug('Computing for model %s...' % model_path)
        aps_test = compute_batch_ap(model, dataset_test, dataset_test.image_ids, config, verbose=1)
        log.info(
            "[{}] Mean mAP over {} tests images: {:.4f} (computed mAP over a range of IoU thresholds [0.5,0.95,0.05])".format(model_path, len(aps_test), np.mean(aps_test)))


if __name__ == "__main__":
    main()
