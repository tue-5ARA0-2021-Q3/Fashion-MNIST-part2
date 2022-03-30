import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config import CFG
# import git
# import datetime

#print(tf.__version__)
"""
Dataset load
"""
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

"""
Preprocess the data
"""
train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

"""
Generate the model
"""
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=CFG.activation),
    tf.keras.layers.Dense(10)
])

"""
Compile the model
"""
model.compile(optimizer=CFG.optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
"""
Tensorboard settings
"""
#
# repo = git.Repo(search_parent_directories=True)
# sha = repo.head.object.hexsha[0: 10]  # Take first 10 characters
# log_dir = "../logs/fit/" + sha + "-" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


"""
Train the model
"""
#history = model.fit(train_images, train_labels, epochs=CFG.epochs, validation_data=(test_images, test_labels), callbacks= [tensorboard_callback])
history = model.fit(train_images, train_labels, epochs=CFG.epochs, validation_data=(test_images, test_labels))
model.save_weights('../models/' + CFG.model_name + '.h5')


if (CFG.debug):
    #Visualize accuracy vs epoch
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("accuracy_epoch.png", dpi=120)
    plt.show()
    plt.close()
    #Visualize loss vs epoch
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("loss_epoch.png", dpi=120)
    plt.show()


"""
Evaluate the model
"""
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
