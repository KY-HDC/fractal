import tensorflow as tf
import tensorflow_datasets as tfds

train_ds = tfds.load('mnist', split='train')
test_ds = tfds.load('mnist', split='test')

def data_normalize(ds):
    return ds.map(lambda sample: {
        'image': tf.cast(sample['image'], tf.float32) / 256.,
        'label': sample['label']
    })
