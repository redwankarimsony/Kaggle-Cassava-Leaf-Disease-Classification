import pandas as pd, json, numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.platform import flags
from PIL import Image
FLAGS = flags.FLAGS

from albumentations import (Compose, JpegCompression, RandomBrightnessContrast, 
                            HueSaturationValue,  HorizontalFlip, ShiftScaleRotate, RGBShift)
AUTOTUNE = tf.data.experimental.AUTOTUNE


# Instantiate augments
# we can apply as many augments we want and adjust the values accordingly
# here I have chosen the augments and their arguments at random
transforms = Compose([
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            RandomBrightnessContrast(p=0.5),
            JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
            RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            HorizontalFlip()])



def aug_fn(image):
    data = {"image":image.astype(np.uint8)}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    aug_img = tf.cast(aug_img/255.0, tf.float32)
    return aug_img

    

def get_train_and_valid_data():
    labeled_datagen = ImageDataGenerator( validation_split=FLAGS.validation_split, 
                                          preprocessing_function=aug_fn)

    dataframe = pd.read_csv(FLAGS.dataframe)
    dataframe['label'] = [str(cls_) for cls_ in dataframe['label'] ]

    train_dataset = labeled_datagen.flow_from_dataframe(dataframe, 
                                                        directory=FLAGS.img_dir,
                                                        x_col= 'image_id',
                                                        y_col = 'label', 
                                                        target_size=(FLAGS.img_width,FLAGS.img_height), 
                                                        batch_size=FLAGS.batch_size,
                                                        shuffle=True,
                                                        color_mode='rgb',
                                                        class_mode='sparse',
                                                        subset='training')
    valid_dataset = labeled_datagen.flow_from_dataframe(dataframe, 
                                                        directory=FLAGS.img_dir,
                                                        x_col= 'image_id',
                                                        y_col = 'label', 
                                                        target_size=(FLAGS.img_width,FLAGS.img_height), 
                                                        batch_size=FLAGS.batch_size,
                                                        shuffle=True, 
                                                        color_mode='rgb',
                                                        class_mode='sparse',
                                                        subset='validation')
                                                                                                                       
    return train_dataset, valid_dataset