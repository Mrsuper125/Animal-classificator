import tensorflow as tf


def configure_for_performance(ds):
    AUTOTUNE = tf.data.AUTOTUNE

    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
