import cv2
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import warnings
warnings.filterwarnings("ignore")


image_exts = ['jpeg', 'png', 'bmp', 'jpg']
data_dir = 'data'
patterns = [os.path.join(data_dir, f'*/*.{ext}') for ext in image_exts ]
image_paths = tf.data.Dataset.list_files(patterns, shuffle=True)

num_images = tf.data.experimental.cardinality(image_paths).numpy()
print(f"Found {num_images} images.")

class_list = []
for item in os.listdir(data_dir):
    if os.path.isdir(os.path.join(data_dir, item)):
        class_list.append(item)
CLASS_NAMES = np.array(class_list)

def process_path(file_path):
    label = tf.strings.split(file_path, os.path.sep)[-2] == CLASS_NAMES
    label = tf.argmax(label)

    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=3)
    img.set_shape([None, None, 3])

    img = tf.image.resize(img, [256, 256])
    img = img / 255.0

    return img, label

labeled_data = image_paths.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

train_size = int(num_images * 0.7)
val_size = int(num_images * 0.2)
test_size = num_images - train_size - val_size

train = labeled_data.take(train_size)
val = labeled_data.skip(train_size).take(val_size)
test = labeled_data.skip(train_size + val_size).take(test_size)

BATCH_SIZE = 32
train = train.batch(BATCH_SIZE)
val = val.batch(BATCH_SIZE)
test = test.batch(BATCH_SIZE)

model = Sequential()
model.add(Conv2D(16,(3,3),1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy','precision', 'recall'])
model.summary()
hist = model.fit(train, epochs=20, validation_data = val)

plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.plot(hist.history['accuracy'], label='accuracy')
plt.plot(hist.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

results = model.evaluate(test, verbose = 1 )
print(f"\n--- Evaluation Results ---")
print(f"Loss: {results[0]:.4f}")
print(f"Accuracy: {results[1]:.4f}")
print(f"Precision: {results[2]:.4f}")
print(f"Recall: {results[3]:.4f}")

model_dir = 'models'
model_path = os.path.join(model_dir, 'image_classifier.keras')
model.save(model_path)

test_image = cv2.imread('img.png')
resize = tf.image.resize(test_image, (256,256))
input_tensor = tf.expand_dims(resize, axis=0)
yhat = model.predict(input_tensor)
if yhat > 0.5:
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')










