import tensorflow as tf
import os
import cv2
import multiprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_fn(proto):
    key_mapping = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    features = tf.io.parse_single_example(proto, key_mapping)
    
    # parse image
    image = features['image']
    image = tf.io.decode_raw(image, tf.float32)
    image = tf.reshape(image, (256, 256, 3))
    image = (image-127.5) / 127.5
    return image, features['label']

def get_dataset(dataset, batch_size=128, parse_fn=parse_fn):
    dataset = get_raw_dataset(dataset, parse_fn)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=512)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(512)
    dataset.cache()
    return dataset

def get_raw_dataset(dataset, parse_fn=parse_fn):
    dataset = tf.data.Dataset.list_files(
        os.path.join(os.environ['HOME'], 'kaggle_data/chest_xray/{}/tfrecords/*'.format(dataset)),
    )
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=multiprocessing.cpu_count(),
        num_parallel_calls=multiprocessing.cpu_count()
    )
    dataset = dataset.map(parse_fn, 
                          num_parallel_calls=multiprocessing.cpu_count()-1)
    return dataset

def get_random_dataset(dataset, batch_size=128, parse_fn=parse_fn):
    # datagen = ImageDataGenerator(
    #     rescale=1. / 255,)
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True)
    
    dataset = get_raw_dataset(dataset, parse_fn)
    
    def map_fn(image, y):
    # path_str is just a normal string here
        numpy_image = image.numpy()
        img_cv = cv2.resize(numpy_image, (128, 128))
        gray_image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray_image = gray_image.reshape(128, 128, 1)
        modified_image = gray_image
        # modified_image = datagen.random_transform(img_cv)
        return modified_image, y
    
    def set_shapes(img, label):
        img.set_shape((batch_size, 128, 128, 1))
        label.set_shape((batch_size, ))
        return img, label

    dataset = dataset.map(lambda x, y: tf.py_function(map_fn, (x, y,), Tout=(tf.float32, tf.int64)), 
                          num_parallel_calls=multiprocessing.cpu_count())
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=512)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(set_shapes)
    dataset = dataset.prefetch(64)
    return dataset
    