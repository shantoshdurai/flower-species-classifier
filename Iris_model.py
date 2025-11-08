import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

from tensorflow.keras.models import Model


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "flower_data",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "flower_data",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
output = GlobalAveragePooling2D()(x)


num_classes = len(train_ds.class_names)
output = Dense(num_classes, activation='softmax')(output)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history = model.fit(train_ds, validation_data=val_ds, epochs=10)


model.save("my_flower_cnn.h5")
