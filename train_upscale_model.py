from diffusion import *
from diffusion.metrics import ESPCNCallback
from diffusion.models import UpscaleModel
from diffusion.processing import Prepare_Dataset
import gc

train_dataset = Prepare_Dataset(train_dataset_path, extension, "upscale")
val_dataset = Prepare_Dataset(val_dataset_path, extension, "upscale")

early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=upscale_checkpoints_path,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

upscale_factor = hr_image_size / image_size
model = UpscaleModel(upscale_factor=upscale_factor, channels=upscale_channels)


class RemoveGarbaseCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


callbacks = [ESPCNCallback(), early_stopping_callback, model_checkpoint_callback, RemoveGarbaseCallback()]
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss=loss_fn,
              )

model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=upscale_num_epochs,
          callbacks=callbacks,
          verbose=2
          )

model.load_weights(upscale_checkpoints_path)
tf.saved_model.save(model, upscale_model_path)
