import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device{}'.format(tf.test.gpu_device_name()))
else:
    print("not using gpu")
