import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# List available devices
print("Physical devices:", tf.config.list_physical_devices())
print("Logical devices:", tf.config.list_logical_devices())

# Check for GPUs
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using GPU.")
else:
    print("TensorFlow is not using GPU.")
