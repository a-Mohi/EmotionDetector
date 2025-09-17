import tensorflow as tf
import cv2
import os

model_dir = 'models'
model_path = os.path.join(model_dir, 'image_classifier.keras')
model = tf.keras.models.load_model(model_path)

test_image = cv2.imread('img.png')
resize = tf.image.resize(test_image, (256,256))
input_tensor = tf.expand_dims(resize, axis=0)
yhat = model.predict(input_tensor)
if yhat > 0.5:
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')
