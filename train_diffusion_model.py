from diffusion import *

from diffusion.processing import Prepare_Dataset
from diffusion.models import DiffusionModel
import gc

train_dataset = Prepare_Dataset(train_dataset_path, extension, "diffusion")
val_dataset = Prepare_Dataset(val_dataset_path, extension, "diffusion")

# create and compile the model
# pixelwise mean absolute error is used as loss
model = DiffusionModel(image_size, widths, block_depth)
model.compile(
    optimizer=keras.optimizers.experimental.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    loss=keras.losses.mean_absolute_error,
)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=diffusion_checkpoints_path,
    save_weights_only=True,
    monitor="val_kid",
    mode="min",
    save_best_only=True,
)


class RemoveGarbaseCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


# calculate mean and variance of training dataset for normalization
model.normalizer.adapt(train_dataset)

model.fit(
    train_dataset,
    epochs=diffusion_num_epochs,
    validation_data=val_dataset,
    callbacks=[
        checkpoint_callback,
        RemoveGarbaseCallback(),
    ],
)

model.load_weights(diffusion_checkpoints_path)
tf.saved_model.save(model, diffusion_model_path, signatures={"generate": model.Generate})
