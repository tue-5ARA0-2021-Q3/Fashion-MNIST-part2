import tensorflow as tf
from src.config import CFG
import matplotlib.pyplot as plt
import datetime
import git
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy

class Learner:
    """ Class to train & save the model """

    def __init__(self):
        self.folder_ml = 'models/'
        self.model = None

    def make_model(self):
        """
        Creates the model
        """
        self.model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation=CFG.activation),
            Dense(10)
        ])
        self.model.compile(
            optimizer=CFG.optimizer,
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    def train_model(self):
        """
        Training the model
        """
        # Dataset load
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        # Preprocess the data
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        # Generate the model
        self.make_model()

        # Tensorboard settings
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha[0: 10]  # Take first 10 characters
        log_dir = "../logs/fit/" + sha + "-" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        print('Fitting model...')
        history = self.model.fit(
            train_images, train_labels, epochs=CFG.epochs,
            validation_data=(test_images, test_labels),
            callbacks=[tensorboard_callback])

        if (CFG.debug):
            # Visualize accuracy vs epoch
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig("accuracy_epoch.png", dpi=120)
            plt.show()
            plt.close()
            # Visualize loss vs epoch
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig("loss_epoch.png", dpi=120)
            plt.show()

        self.evaluate_model(test_images, test_labels)

    def save_model(self, model_name):
        """
        Saving the model using pickle or h5
        :param model_name: name of model
        :type model_name: str
        """
        filename = CFG.model_name + model_name + '.h5'
        self.model.save_weights(self.folder_ml + filename)

    def evaluate_model(self, test_images, test_labels):
        """
        Saving the model using pickle or h5
        :param test_images:
        :param test_labels:
        """
        test_loss, test_acc = self.model.evaluate(test_images, test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)