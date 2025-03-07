import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 20
DATASET_DIR = r'C:\Users\pc\Downloads\homework\dishes'

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

CATEGORIES = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
if not CATEGORIES:
    raise FileNotFoundError(f"No category folders found in {DATASET_DIR}")

NUM_CLASSES = len(CATEGORIES)
print(f"Found {NUM_CLASSES} categories: {CATEGORIES}")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

model.save('dishes_recognition_model.h5')

def predict_and_display(model, image_path, categories):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    prediction = model.predict(img_array)
    predicted_class = categories[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    return img, predicted_class, confidence

available_images = []
for category in CATEGORIES:
    category_path = os.path.join(DATASET_DIR, category)
    images = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if images:
        for img in images:
            available_images.append((category, os.path.join(category_path, img)))

if not available_images:
    raise FileNotFoundError("No images found in any category folder")

plt.figure(figsize=(15, 10))
for i in range(min(6, len(available_images))):
    category, image_path = np.random.choice(available_images)
    img, predicted_class, confidence = predict_and_display(model, image_path, CATEGORIES)
    
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2f}%')
    plt.axis('off')

plt.tight_layout()
plt.show()