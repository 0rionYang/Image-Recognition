import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# RGB values between 0~255
# feature scaling
x_train = x_train / 255.0  
x_test = x_test / 255.0



# reshape, activation function: softmax
"""model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
"""

# CNN
model = keras.Sequential([
    keras.layers.Conv2D(28,(3,3),activation = 'relu', input_shape=(28, 28,1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),                         
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
model.summary()





# Optimization，learning rate = 0。1
optimizer = tf.keras.optimizers.SGD(0.1)


# training the data
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])

'''model.compile(optimizer=tf.keras.optimizers.SGD(0.1),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])'''

model.fit(x_train,y_train,epochs=5,batch_size=256)

# test error/ accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Acc:',test_acc)

