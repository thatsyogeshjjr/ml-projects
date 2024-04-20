import tensorflow as tf

model = tf.keras.models.load_model('handwritten_digits.model.keras')
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


loss, accuracy = model.evaluate(x_test, y_test)

print("loss:\t", loss)
print("accuracy:\t", accuracy)
