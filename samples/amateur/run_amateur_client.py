import cv2 as cv2
import requests
import json
import numpy as np
import os
# initialize the Keras REST API endpoint URL along with the input image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
BASE_DIR = 'F:/AMATEUR/fichiers_pour_tests/'
RESULT_DIR = 'F:/AMATEUR/results/'
# curl -X POST -F image=@IMG_20190216_121857.jpg 'http://localhost:5000/predict'


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main():
    images = [f for f in os.listdir(BASE_DIR) if f.endswith('.jpg')]
    for image_file in images:
        # load the input image and construct the payload for the request
        image = open(BASE_DIR+image_file, "rb").read()
        payload = {"image": image}

        # submit the request
        r = requests.post(KERAS_REST_API_URL, files=payload).json()

        # ensure the request was sucessful
        if r["success"]:
            im = np.asarray(json.loads(r['image']))
            assert isinstance(im, np.ndarray)
            cv2.imwrite(RESULT_DIR+image_file,im)

        # otherwise, the request failed
        else:
            print("Request failed")


if __name__ == "__main__":
    main()