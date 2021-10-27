import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

from tensorflow import keras

#テンソル化
def convert_img(file): 
    global h,w
    image = tf.io.read_file(file)
    image = tf.image.decode_jpeg(image, channels=3)

    #4次元配列（4階テンソル)化にreshapeするためPIL化
    pilImg_rgb = Image.fromarray(np.uint8(image))
    pilImg_rgb = np.array(pilImg_rgb, dtype='float32')
    h = len(image)
    w = len(image[0])
    a = pilImg_rgb.reshape(1, h, w, 3)

    #画像の数値をrgb用からfloat用にする（255で割って0～1の間にする）
    image = tf.convert_to_tensor(a/255.0)

    #大き目の画像サイズ縮小（元画像の3分の１)
    if h >= 1000 or w >= 1000:
        h = int(h/3)
        w = int(w/3)
        image = tf.image.resize(image, [h, w])
    return image

#content_image = convert_img("bike2-sen.jpg")
#style_image = convert_img("dragonball.jpg")

def digit(c,s):
    style_image = tf.nn.avg_pool(s, ksize=[3,3], strides=[1,1], padding='SAME')
    hub_module = keras.models.load_model("model")

    outputs = hub_module(tf.constant(c), tf.constant(style_image))
    stylized_image = outputs[0]
    return stylized_image

