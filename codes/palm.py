import tensorflow as tf
import pathlib
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


DATASET_PATH = r"C:\Users\pc\Downloads\homework\palm_dataset"
data_dir = pathlib.Path(DATASET_PATH)


img_paths = list(data_dir.glob("*.*")) 

if len(img_paths) == 0:
    raise ValueError("Thư mục không có ảnh hoặc sai đường dẫn.")

print(f"Tìm thấy {len(img_paths)} ảnh.")


def load_and_preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return img_array


image_data = np.array([load_and_preprocess_image(str(p)) for p in img_paths], dtype=np.float32)


labels = np.zeros(len(image_data)) 


dataset = tf.data.Dataset.from_tensor_slices((image_data, labels)).batch(4)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

EPOCHS = 10
model.fit(dataset, epochs=EPOCHS)


random_img_path = random.choice(img_paths)
img = tf.keras.preprocessing.image.load_img(random_img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

plt.imshow(img)
plt.title(f'Prediction: {prediction[0][0]:.4f}')
plt.axis("off")
plt.show()
model.save("palm_model.h5")