from diffusion import *

from keras.utils import save_img, img_to_array
from os import makedirs, path
from keras.utils import load_img, img_to_array
import PIL


def Upscale_Image(model, img_path: str):
    """Predict the result based on input image and restore the image as RGB."""
    in_img = load_img(img_path)
    in_img = in_img.resize(
        (image_size, image_size),
        PIL.Image.BICUBIC,
    )

    # Convert to YUV color format
    ycbcr = in_img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0

    input_array = np.expand_dims(y, axis=0)
    output = model(input_array)
    out_img_y = output[0]

    # Restore the image in RGB color space.
    out_img_y = tf.clip_by_value(out_img_y * 255.0, 0.0, 255.0)
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y[:, :, 0]), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )
    return out_img


diffusion_model = tf.saved_model.load(diffusion_model_path)
upscale_model = tf.saved_model.load(upscale_model_path)

generated_images = diffusion_model.generate()
generated_images = tf.clip_by_value(generated_images * 255.0, 0.0, 255.0)
generated_images = tf.cast(generated_images, tf.uint8)

if not path.exists(generated_output):
    makedirs(generated_output)

for (index, image) in enumerate(generated_images):
    array_image = img_to_array(image)
    save_img(generated_output + str(index) + "_lr" + extension, array_image)
    hr_image = Upscale_Image(upscale_model, generated_output + str(index) + "_lr" + extension)
    hr_image.save(generated_output + str(index) + "_hr" + extension)
