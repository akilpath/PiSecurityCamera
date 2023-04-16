# import tensorflow as tf
#
# converter = tf.lite.TFLiteConverter.from_saved_model("./")
# tfLiteModel = converter.convert()
# with open('model20230415.tflite', 'wb') as file:
#     file.write(tfLiteModel)
# print('done')

import pathlib
import tensorflow as tf

# image height x width
IMAGE_SIZE = (480, 640)
# batch size
BATCH_SIZE = 16

datasetDir = "./dataset"
training_ds = tf.keras.utils.image_dataset_from_directory(datasetDir,
                                                          validation_split=0.2,
                                                          subset="training",
                                                          seed=123,
                                                          image_size=IMAGE_SIZE,
                                                          batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(datasetDir,
                                                     validation_split=0.2,
                                                     subset="validation",
                                                     seed=123,
                                                     image_size=IMAGE_SIZE,
                                                     batch_size=BATCH_SIZE)

classNames = training_ds.class_names
numOfClasses = len(classNames)

training_ds = training_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
model = tf.keras.models.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.Rescaling(1. / 255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=["accuracy"])
#model.summary()
epochs = 20
history = model.fit(
    training_ds,
    validation_data = val_ds,
    epochs = epochs
)

if input() == "y":
    tf.saved_model.save(model, "./")
