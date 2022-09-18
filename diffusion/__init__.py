import math
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from keras import layers

import numpy as np
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Diffusion parameters
import tensorflow.lite as lit
# CUDA path
cuda_path = "/opt/cuda"
os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir=" + cuda_path
# data
train_dataset_path = "nebulae_data/train"
val_dataset_path = "nebulae_data/validation"
extension = ".jpeg"
diffusion_checkpoints_path = "diffusion_checkpoints/diffusion_model"
diffusion_model_path = "diffusion_model"
generated_output = "generated_output/"
dataset_repetitions = 5
diffusion_num_epochs = 1  # train for at least 50 epochs for good results
image_size = 256
hr_image_size = 1280
# KID = Kernel Inception Distance, see related section
kid_image_size = 75
kid_diffusion_steps = 20
shuffle_times = 10

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2

# optimization
batch_size = 16
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4


# Upscale parameters

upscale_checkpoints_path = "upscale_checkpoints/upscale_model"
upscale_model_path = "upscale_model"

upscale_batch_size = 4
upscale_channels = 1
upscale_num_epochs = 1
