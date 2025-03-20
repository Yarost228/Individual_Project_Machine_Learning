import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Отключаем лишние логи

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import signal
import sys

# Определяем параметры
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 28
TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'
MODEL_PATH = 'model/NeyronchikBeter.h5'
BEST_MODEL_PATH = 'model/NeyronchikBeter_best.h5'

# Создаем директории, если они не существуют
os.makedirs('model', exist_ok=True)
os.makedirs('dataset/train', exist_ok=True)
os.makedirs('dataset/test', exist_ok=True)

# Создаем поддиректории для каждого класса
for gesture in ['rock', 'paper', 'scissors', 'unknown']:
    os.makedirs(f'dataset/train/{gesture}', exist_ok=True)
    os.makedirs(f'dataset/test/{gesture}', exist_ok=True)

# Создаем генераторы данных с аугментацией
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Загружаем данные
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Создаем модель
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Создаем callback для сохранения лучшей модели
checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

try:
    # Обучаем модель
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator,
        callbacks=[checkpoint]
    )

    # Сохраняем финальную модель
    model.save(MODEL_PATH)
    print(f"\nФинальная модель сохранена в {MODEL_PATH}")

    # Выводим результаты обучения
    print("\nРезультаты обучения:")
    print(f"Точность на тренировочных данных: {history.history['accuracy'][-1]:.4f}")
    print(f"Точность на тестовых данных: {history.history['val_accuracy'][-1]:.4f}")

except KeyboardInterrupt:
    print("\nПрерывание обучения. Сохраняем модель...")
    model.save(MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")
    sys.exit(0)
