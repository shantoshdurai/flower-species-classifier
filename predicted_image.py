from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os


model = load_model("my_flower_cnn.h5")


img_path = "HANN-702.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)


pred = model.predict(img_array)

pred_idx = np.argmax(pred, axis=1)[0]


class_names = sorted(os.listdir("flower_data"))

print("Predicted Species:", class_names[pred_idx])
