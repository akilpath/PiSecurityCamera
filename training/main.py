import tensorflow as tf
import matplotlib.pyplot as plt

# image height x width
IMAGE_SIZE = (500, 500)
# batch size
BATCH_SIZE = 64

datasetDir = "..//dataset"
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
    tf.keras.layers.RandomBrightness(0.3, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    tf.keras.layers.RandomContrast(0.3),
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(500, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])



model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()
epochs = 6
history = model.fit(
    training_ds,
    validation_data=val_ds,
    epochs=epochs
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

fig = plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='upper left')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

fig.savefig("training_accuracy.png", dpi=200)

print("Save Model?")
if input() == "y":
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tfLiteModel = converter.convert()
    print("Converted to tflite obj")
    with open('model3.tflite', 'wb') as file:
        file.write(tfLiteModel)
    print('done')
