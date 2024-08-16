import tensorflow as tf
import tensorflow_datasets as tfds
from protocol_args import batch_size

def data_normalize(ds):
    return ds.map(lambda sample: {
        'image': tf.cast(sample['image'], tf.float32) / 255.,
        'label': sample['label']
    })


def prepare_dataset(batch_size, seed=42):
    
    # Load and normalize
    train_ds = tfds.load('mnist', split='train')
    test_ds = tfds.load('mnist', split='test')
    train_ds = data_normalize(train_ds).shuffle(buffer_size=10, seed=seed).batch(batch_size)
    test_ds = data_normalize(test_ds).shuffle(buffer_size=10, seed=seed).batch(batch_size)
    
    # Info
    total_batch = train_ds.cardinality().numpy()
    total_tbatch = test_ds.cardinality().numpy()
    info = {'train_batch': total_batch, 'test_batch': total_tbatch}
    return tfds.as_numpy(train_ds), tfds.as_numpy(test_ds), info
    
# Prepare dataset
train_ds, test_ds, info = prepare_dataset(batch_size)
batch = next(iter(train_ds))    # a batch for initializing
x, y = batch['image'], batch['label']


if __name__ == "__main__":
    ds = prepare_dataset(100)
    
