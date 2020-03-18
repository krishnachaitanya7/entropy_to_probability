import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
import numpy as np
import datetime
import shutil
import os

global model
# changed_lr = False
folder_name = "my_model"


def save_model(fl_name):
    global model
    if os.path.isdir(fl_name):
        shutil.rmtree(fl_name)
        os.mkdir(fl_name)
    model.save(model, save_format='tf')


class my_model(tf.keras.Model):
    def __init__(self):
        super(my_model, self).__init__()
        self.l1(layers.SimpleRNN(5, return_sequences=True))
        self.l2(layers.SimpleRNN(5))
        self.l3(layers.Dense(1, activation=tf.keras.activations.sigmoid))

    def call(self, inputs):
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model
        # https://www.tensorflow.org/guide/keras/custom_layers_and_models
        # You gotta overwrite this call function so that you can implement the forward pass
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        return x


class model_saver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        save_after_epochs = 100  # We will be saving after every these many no of epochs
        if epoch % save_after_epochs == 0:
            save_model(folder_name)


class LearningRateReducerCb(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['loss'] < 1e-04:
            model.stop_training = True


def make_model(fl_name):
    global model
    # First find if there is already a saved model from which you can load on
    if len(os.listdir(fl_name)) != 0:
        # Means we have a saved model, lets load it boys
        print("Saved Checkpoints found, loading it...")
        model = tf.keras.models.load_model(fl_name)
    else:
        # Model ain't present, you have to do it from start
        print("No checkpoints found. Loading from scratch...")
        # model = keras.Sequential()
        # model.add(layers.SimpleRNN(5, return_sequences=True))
        # model.add(layers.SimpleRNN(5))
        # model.add(layers.Dense(1, activation=tf.keras.activations.sigmoid))
        # model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
        #               optimizer=tf.keras.optimizers.Nadam(1e-4),
        #               metrics=['mae'])
        model = my_model()


if __name__ == "__main__":
    df = pd.read_csv("entropy.csv")
    X = df["Input"].values
    y = df["Output"].values
    file_name = "Final_values.csv"
    MAX_EPOCHS = 500
    BATCH = 100
    X_train, X_testing, y_train, y_testing = train_test_split(X, y, test_size=0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_testing, y_testing, test_size=0.5)
    # Change the shape of all
    X_train = X_train.reshape((len(X_train), 1, 1))
    X_test = X_test.reshape((len(X_test), 1, 1))
    y_train = y_train.reshape((len(y_train), 1, 1))
    y_test = y_test.reshape((len(y_test), 1, 1))
    X_val = X_val.reshape((len(X_val), 1, 1))
    y_val = y_val.reshape((len(y_val), 1, 1))
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    make_model(folder_name)
    model.fit(X_train, y_train, epochs=MAX_EPOCHS, batch_size=BATCH, verbose=1,
              callbacks=[LearningRateReducerCb(), tensorboard_callback, model_saver()],
              validation_data=(X_test, y_test))
    y_pred = model.predict(X_val)
    rmse_calculated = mean_squared_error(y_pred.flatten(), y_val.flatten())
    print(f"RMSE on Validation: {rmse_calculated}")
    final_save_matrix = np.stack((X_val.flatten(), y_val.flatten(), y_pred.flatten()), axis=1)
    open(file_name, 'w').close()
    np.savetxt(file_name, final_save_matrix,
               delimiter=',', header="X_validation,Original_Y,Predicted_y", comments="")
    # os.remove(file_name)
    save_model(folder_name)
    # Noted RMSE: 4.1091989895786396e-06
