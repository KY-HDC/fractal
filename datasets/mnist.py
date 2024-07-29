import tensorflow as tf
import tensorflow_datasets as tfds


def data_normalize(ds):
    return ds.map(lambda sample: {
        'image': tf.cast(sample['image'], tf.float32) / 255.,
        'label': sample['label']
    })

def prepare_dataset(batch_size, seed=42, take_size=10):
    train_ds = tfds.load('mnist', split='train')
    test_ds = tfds.load('mnist', split='test')
    # train_ds = data_normalize(train_ds).cache().shuffle(buffer_size=10, seed=seed).batch(batch_size).prefetch(1)
    # test_ds = data_normalize(test_ds).cache().shuffle(buffer_size=10, seed=seed).batch(batch_size).prefetch(1)
    train_ds = data_normalize(train_ds).shuffle(buffer_size=10, seed=seed).batch(batch_size).take(take_size).prefetch(1)
    test_ds = data_normalize(test_ds).batch(batch_size).take(take_size).prefetch(1)
    return train_ds, test_ds


if __name__ == "__main__":
    ds = prepare_dataset(100)