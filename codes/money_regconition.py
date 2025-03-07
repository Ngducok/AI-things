import os 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import shutil

IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = r'C:\Users\pc\Downloads\homework\datasets\money_dataset'

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError("Dataset not found, please check the path")

TEMP_DIR = os.path.join(os.path.dirname(DATASET_DIR), 'temp_dataset')
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR)

CLASS_DIR = os.path.join(TEMP_DIR, 'money')
os.makedirs(CLASS_DIR)

for img in os.listdir(DATASET_DIR):
    if img.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif')):
        src = os.path.join(DATASET_DIR, img)
        dst = os.path.join(CLASS_DIR, img)
        shutil.copy2(src, dst)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2, 
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    TEMP_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    TEMP_DIR, 
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

def predict_and_display(model, image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    prediction = model.predict(img_array)
    confidence = prediction[0][0] * 100
    predicted_class = "Money" if prediction[0][0] > 0.5 else "Not Money"
    
    return img, predicted_class, confidence

plt.figure(figsize=(15, 10))
for i in range(6):
    random_image = np.random.choice(os.listdir(CLASS_DIR))
    image_path = os.path.join(CLASS_DIR, random_image)
    
    img, predicted_class, confidence = predict_and_display(model, image_path)
    
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2f}%')
    plt.axis('off')

plt.tight_layout()
plt.show()

model.save('Money_recognition.h5')
print("Model saved successfully")