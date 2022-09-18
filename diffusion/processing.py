from diffusion import *


def Process_Path(file_path, image_height=hr_image_size, image_width=hr_image_size):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_height, image_width], antialias=True)
    image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
    return image


def Prepare_Dataset(folder_path: str, ext: str, alg_type: str):
    # the validation dataset is shuffled as well, because data order matters
    # for the KID estimation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
    ])
    ds = tf.data.Dataset.list_files(folder_path + "/*" + ext)
    ds = ds.map(Process_Path).map(lambda x: (data_augmentation(x, training=True)), num_parallel_calls=tf.data.AUTOTUNE)
    if alg_type == "diffusion":
        ds = ds.map(lambda x: tf.image.resize(x, [image_size, image_size], antialias=True))
        local_batch_size = batch_size
    if alg_type == "upscale":
        ds = ds.map(lambda x: (Process_Input_Upscale(x), Process_Target_Upscale(x)))
        local_batch_size = upscale_batch_size
    return (ds.cache()
            .repeat(dataset_repetitions)
            .shuffle(shuffle_times * batch_size)
            .batch(local_batch_size, drop_remainder=True)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            )


def Process_Input_Upscale(input_image, image_height=image_size, image_width=image_size):
    input_image = tf.image.rgb_to_yuv(input_image)
    last_dimension_axis = len(input_image.shape) - 1
    y, u, v = tf.split(input_image, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [image_height, image_width], method="area")


def Process_Target_Upscale(input_image):
    input_image = tf.image.rgb_to_yuv(input_image)
    last_dimension_axis = len(input_image.shape) - 1
    y, u, v = tf.split(input_image, 3, axis=last_dimension_axis)
    return y
