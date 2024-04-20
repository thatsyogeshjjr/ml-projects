import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model('handwritten_digits.model.keras')

path = './test images'
files = os.listdir(path)

for file in files:

    img = cv2.imread(f"{path}/{file}")[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"Digit prediction: {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
