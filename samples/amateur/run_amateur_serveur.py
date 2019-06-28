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

os.environ['basedir_a'] = 'F:/Temp2/'
import PIL as PIL
import logging
import mrcnn.model as modellib
import numpy as np
import scipy as scipy
import time
import luccauchon.data.__MYENV__ as E  # Charge en mémoire des variables d'environnement.

E.APPLICATION_LOG_LEVEL = logging.DEBUG
log = E.setup_logger(logger_name=__name__, _level=E.APPLICATION_LOG_LEVEL)

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
MODEL_DIR = 'F:/AMATEUR/models_mask_rcnn/hq20190528T1316/'
MODEL_DIR = 'F:/AMATEUR/models_mask_rcnn/hq20190613T1330/'

import flask
import io
from mrcnn.config import Config
from mrcnn import visualize
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
from flask import send_file
import io
import json
import cv2 as cv2


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


class TestClass:
    def __init__(self):
        inference_config = InferenceConfig()

        # Recreate the model in inference mode
        self.model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

        # Get path to saved weights
        model_path = MODEL_DIR + '/mask_rcnn_hq_0044.h5'
        model_path = MODEL_DIR + '/mask_rcnn_hq_0098.h5'

        # Load trained weights
        self.model.load_weights(model_path, by_name=True)

    def detect(self, images, verbose=0):
        return self.model.detect(images, verbose)


# initialize our Flask application
app = flask.Flask(__name__)
my_model = None


'''class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)'''

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,            np.int16, np.int32, np.int64, np.uint8,            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,             np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            t1 = time.time()
            image = flask.request.files["image"].read()
            ima = cv2.imdecode(np.fromstring(image, dtype=np.uint8), cv2.IMREAD_COLOR)
            #image = Image.open(io.BytesIO(image))
            #ima = np.asarray(image)
            t2 = time.time()
            global graph
            with graph.as_default():
                results = my_model.detect([ima], verbose=0)
            assert 1 == len(results)
            r = results[0]
            t3 = time.time()
            my_result = visualize.display_instances(ima, r['rois'], r['masks'], r['class_ids'], ['BG', 'pt', 'tf'], r['scores'], only_return_image=True)
            assert isinstance(my_result, np.ndarray)
            #my_image = PIL.Image.fromarray(my_result, mode='RGB')
            #my_image = my_image.resize((640,480))
            my_image = cv2.resize(my_result, (1024,768), interpolation=cv2.INTER_AREA)
            t4 = time.time()
            data["success"] = True
            data["image"] = json.dumps(np.asarray(my_image), cls=NumpyEncoder)
            t5 = time.time()
            log.debug('Time to load image: ' + str(t2 - t1) + ' seconds.')
            log.debug('Time to make inference: ' + str(t3 - t2) + ' seconds.')
            log.debug('Time to make output file: ' + str(t4 - t3) + ' seconds.')
            log.debug('Time to make json: ' + str(t5 - t4) + ' seconds.')

    # return flask.send_file(io.BytesIO(my_result), attachment_filename='result.jpg', mimetype='image/jpeg',as_attachment=True)
    return flask.jsonify(data)


def load_model():
    global my_model
    my_model = TestClass()


if __name__ == "__main__":
    load_model()
    graph = tf.get_default_graph()
    app.run()
