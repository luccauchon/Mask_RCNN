import numpy as np
import matplotlib.pyplot as plt
import PIL as PIL
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import threading
from queue import Queue
import traceback
from zipfile import ZipFile
import time
import scipy
from PIL import ImageEnhance
from PIL import ImageFilter
import ntpath
import re
import itertools


print_debug = False

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

import keras
# set the modified tf session as backend in keras
my_sess = get_session()
keras.backend.tensorflow_backend.set_session(my_sess)
import keras as K
from keras import backend as KBE

nb_channel = 3
typeimg = '5xrgbimage'
imgW = 75 * 5
imgH = 75

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
           'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
           'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0',
           '1', '2', '3', '4', '5', '6', '7', '8', '9']
maxTextLen = 5
input_length_for_ctc = 64
O_CTC = 0


def CTCLambdaFunc(args):
    y_pred, y_true, input_length, label_length = args
    # the O_CTC is critical here since the first couple outputs of the RNN tend to be garbage:
    y_pred = y_pred[:, 0:, :]


def LoadImagesFromZip(imgPath, zipFilePath):
    zipfile = ZipFile(zipFilePath)
    files = [name[1] for name in enumerate(zipfile.namelist()) if re.match(imgPath + '/\w+\.jpg', name[1]) is not None]
    return files, zipfile


def loadData(zip_file_path, subsetPercentage=None):
    trainFiles, trainZipFile = LoadImagesFromZip('train', zip_file_path)
    valFiles, valZipFile = LoadImagesFromZip('valid', zip_file_path)
    testFiles, testZipFile = LoadImagesFromZip('test', zip_file_path)
    if 0 == len(testFiles):
        print('  * Using images for TEST from zipfile ' + zip_file_path + '. Have them ready in [basedir_images].')
        testFiles, testZipFile = LoadImagesFromZip('test_real_all', zip_file_path)
    elif subsetPercentage is not None:
        indices = random.sample(range(len(trainFiles)), int(subsetPercentage * len(trainFiles)))
        trainFiles = [trainFiles[i] for i in sorted(indices)]

        indices = random.sample(range(len(valFiles)), int(subsetPercentage * len(valFiles)))
        valFiles = [valFiles[i] for i in sorted(indices)]

        indices = random.sample(range(len(testFiles)), int(subsetPercentage * len(testFiles)))
        testFiles = [testFiles[i] for i in sorted(indices)]

    imagecount_train = len(trainFiles)
    imagecount_val = len(valFiles)
    imagecount_test = len(testFiles)
    print('Using ' + str(imagecount_train) + ' files for training, ' + str(imagecount_val) + ' for validation and ' + str(imagecount_test) + ' for testing.')
    return trainFiles, trainZipFile, valFiles, valZipFile, testFiles, testZipFile


g_lock = threading.Lock()


def NextSample(n, indexes, currIndex, imgFiles):
    if currIndex >= n: # This will never be true in multithread.
        currIndex = 0
        random.shuffle(indexes)
    return imgFiles[indexes[currIndex]], currIndex + 1


def find_hsv(img):
    r,g,b = img.split()
    Hdat = []
    Sdat = []
    Vdat = []
    import colorsys
    for rd,gn,bl in zip(r.getdata(),g.getdata(),b.getdata()) :
        h,s,v = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)
        Hdat.append(int(h*255.))
        Sdat.append(int(s*255.))
        Vdat.append(int(v*255.))
    return [np.asarray(Hdat),np.asarray(Sdat),np.asarray(Vdat)]


def create5xLargeRGBImage(img_file, imgW, imgH):
    img = PIL.Image.open(img_file).resize((imgW // 5, imgH), PIL.Image.ANTIALIAS)
    h, s, v = find_hsv(img)
    mv = 0.85 * v.mean()
    img = img.convert("L")
    w1 = img.point(lambda x: 32 if x < mv else 128)
    w2 = ImageEnhance.Brightness(w1).enhance(0.33)
    w3 = ImageEnhance.Brightness(w1).enhance(3)
    w4 = ImageEnhance.Contrast(w1).enhance(2 * 3)
    w5 = ImageEnhance.Sharpness(w1).enhance(2 * 3)

    new_im = PIL.Image.new('L', (imgW, imgH))

    x_offset = 0
    for im in [w1, w2, w3, w4, w5]:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    new_im = new_im.convert('RGB')
    assert isinstance(new_im, PIL.Image.Image)
    assert new_im.width == imgW, '(~W) ' + str(new_im.width()) + ' == ' + str(imgW)
    assert new_im.height == imgH, '(~H) ' + str(new_im.height()) + ' == ' + str(imgH)
    return new_im


def ReadConvertImage(imgPath, basedir_images, imgW, imgH, conv, nb_channel):
    img_file = basedir_images + imgPath
    # Read the specified file.
    if conv == 'RGB':
        assert 3 == nb_channel, '3!=' + str(nb_channel)
        img = scipy.misc.imread(name=img_file, mode='RGB')
        img = scipy.misc.imresize(img, (imgH, imgW), interp='nearest')
        img = np.moveaxis(img, 1, 0)
        img = img.astype(np.float32)
        img /= 255.0
    elif conv == '1ximage':
        assert 1 == nb_channel, '1!=' + str(nb_channel)
        img = create1xImage(img_file, imgW, imgH)
        img = np.asarray(img)
    elif conv == '1xrgbimage':
        assert 3 == nb_channel, '3!=' + str(nb_channel)
        img = create1xRGBImage(img_file, imgW, imgH)
        img = np.asarray(img)
    elif conv == '5xrgbimage':
        assert 3 == nb_channel, '3!=' + str(nb_channel)
        img = create5xLargeRGBImage(img_file, imgW, imgH)
        img = np.asarray(img)
        img = np.moveaxis(img, 1, 0)
    elif conv == '5xLimage':
        assert 1 == nb_channel, '1!=' + str(nb_channel)
        img = create5xLargeImage(img_file, imgW, imgH)
        img = np.asarray(img)
    else:
        assert False

    assert imgW == img.shape[0], '(W)' + str(imgW) + ' == ' + str(img.shape[0]) + '  img.shape=' + str(img.shape)
    assert imgH == img.shape[1], '(H)' + str(imgH) + ' == ' + str(img.shape[1]) + '  img.shape=' + str(img.shape)

    text = ntpath.basename(os.path.splitext(img_file)[0])
    if '_' in text:
        text = text.split('_')[0]

    return img, text


def TextToLabels(text,letters):
    return list(map(lambda x: letters.index(x), text))


def NextBatchAsync(n, batchSize, imgPaths, basedir_images, imgW, imgH, nb_channel, maxTextLen, typeimg,
                   input_length_for_ctc, O_CTC, letters, nb_workers):
    qData = Queue()
    qCurrIndex = Queue()
    qCurrIndex.put(0)

    indexes = list(range(n))
    random.shuffle(indexes)

    def worker():
        while True:
            start = time.clock()
            try:
                if qData.qsize() > 32:
                    time.sleep(1)
                    continue

                assert not KBE.image_data_format() == 'channels_first'

                X_data = np.ones([batchSize, imgW, imgH, nb_channel])
                Y_data = np.ones([batchSize, maxTextLen])
                input_length = np.ones((batchSize, 1)) * (input_length_for_ctc - O_CTC)
                label_length = np.zeros((batchSize, 1))

                startIndex = qCurrIndex.get()
                if startIndex >= n or startIndex + batchSize >= n:
                    startIndex = 0
                    random.shuffle(indexes)
                qCurrIndex.put(startIndex + batchSize)

                for i in range(batchSize):
                    imgFile, startIndex = NextSample(n, indexes, startIndex, imgPaths)
                    img, text = ReadConvertImage(imgFile, basedir_images, imgW, imgH, typeimg, nb_channel)
                    if typeimg == 'RGB' or typeimg == '1xrgbimage' or typeimg == '5xrgbimage':
                        assert 3 == nb_channel
                        X_data[i] = img
                    else:
                        assert 1 == nb_channel
                        X_data[i] = np.expand_dims(img, -1)
                    Y_data[i] = TextToLabels(text, letters)
                    label_length[i] = len(text)
                    assert maxTextLen == len(Y_data[i])
                    assert 5 == len(text)

                inputs = {
                    'the_input': X_data,
                    'the_labels': Y_data,
                    'input_length': input_length,
                    'label_length': label_length,
                }
                outputs = {'ctc': np.zeros([batchSize])}
                qData.put((inputs, outputs))
            except Exception:
                with g_lock:
                    print('[' + threading.current_thread().name + '] Error in generator.')
                with g_lock:
                    traceback.print_exc()
                pass

            end = time.clock()
            with g_lock:
                if print_debug:
                    print('[' + threading.current_thread().name + ']Preparing batch of ' + str(batchSize) + ' elements in ' + str((end - start)) + ' seconds.')

    # Create the thread pool.
    for i in range(nb_workers):
        t = threading.Thread(target=worker)
        t.daemon = False
        t.start()

    while True:
        start = time.clock()
        inputs, outputs = qData.get()
        end = time.clock()
        with g_lock:
            if print_debug:
                print('[' + threading.current_thread().name + ']Getting batch of ' + str(batchSize) + ' elements in ' + str((end - start)))
        yield (inputs, outputs)

def decode_batch(out, O_CTC, letters):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, O_CTC:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret

basedir_images = 'F:/AMATEUR/LCLCL_images_pour_test/'
trainFiles, trainZipFile, valFiles, valZipFile, testFiles, testZipFile = loadData(zip_file_path='F:/AMATEUR/LCLCL_images_pour_test/LCLCL__real_all.zip')

model_to_reload = 'F:/AMATEUR/lclcl_models/gen20_2000000_DensenNet201_5xrgbimage_375x75x3_weights.18-0.0457.hdf5'
custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred}
start = time.clock()
trained = keras.models.load_model(model_to_reload, custom_objects=custom_objects)
end = time.clock()
print(end-start)
batch_size_test = 8
imagecount_test = len(testFiles) # Number of images to process.
name_in='the_input'
name_out='softmax3'
net_inp = trained.get_layer(name=name_in).input
net_out = trained.get_layer(name=name_out).output
for inp_value, _ in NextBatchAsync(imagecount_test, batch_size_test, testFiles, basedir_images, imgW, imgH, nb_channel, maxTextLen, typeimg, input_length_for_ctc, O_CTC,letters,nb_workers=4):
    start = time.clock()
    bs = inp_value['the_input'].shape[0]
    assert bs == batch_size_test

    X_data = inp_value['the_input']
    net_out_value = my_sess.run(net_out, feed_dict={net_inp:X_data})
    Y_pred_text = decode_batch(net_out_value,O_CTC,letters)
    Y_true = inp_value['the_labels']

    end = time.clock()
    print(end - start)