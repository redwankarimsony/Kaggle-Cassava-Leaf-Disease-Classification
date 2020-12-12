import tensorflow as tf
from tensorflow import keras
from keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
# from keras.applications import EfficentNetB0 
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def __get_backbone():
    if FLAGS.model =='DenseNet121':
        return DenseNet121(include_top= False, 
                    weights='imagenet',
                    input_shape=(FLAGS.img_width, FLAGS.img_height,3))
    elif FLAGS.model == 'DenseNet169':
        return DenseNet169(include_top= False, 
                    weights='imagenet',
                    input_shape=(FLAGS.img_width, FLAGS.img_height,3))
    elif FLAGS.model == 'DenseNet201':
        return DenseNet201(include_top= False, 
                    weights='imagenet',
                    input_shape=(FLAGS.img_width, FLAGS.img_height,3),)
    elif FLAGS.model == 'ResNet50':
        return ResNet50(include_top= False, 
                    weights='imagenet',
                    input_shape=(FLAGS.img_width, FLAGS.g_height,3),)


def get_model():
    backbone = __get_backbone()
    backbone.trainable = False

    model = keras.Sequential([backbone,
                            GlobalAveragePooling2D(),
                            Dense(FLAGS.no_of_classes,activation='softmax')])
    return model

