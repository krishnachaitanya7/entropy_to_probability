"""
Refernces: https://lambdalabs.com/blog/tensorflow-2-0-tutorial-03-saving-checkpoints/
"""
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

# changed_lr = False
folder_name = "my_model"


def save_model(fl_name):
    global model
    if os.path.isdir(fl_name):
        shutil.rmtree(fl_name)
        os.mkdir(fl_name)
    model.save(fl_name, save_format='tf')


class LearningRateReducerCb(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['loss'] < 1e-04:
            model.stop_training = True


def make_model():
    my_model = keras.Sequential()
    my_model.add(layers.SimpleRNN(5, return_sequences=True))
    my_model.add(layers.SimpleRNN(5))
    my_model.add(layers.Dense(1, activation=tf.keras.activations.sigmoid))
    my_model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                  optimizer=tf.keras.optimizers.Nadam(1e-4),
                  metrics=['mae'])
    return my_model


if __name__ == "__main__":
    df = pd.read_csv("entropy.csv")
    X = df["Input"].values
    y = df["Output"].values
    file_name = "Final_values.csv"
    MAX_EPOCHS = 2000
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
    # Now let's code the callbacks
    # Tensorboard callbacks
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # Model saving Checkpoint callback
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1, period=10)
    model = make_model()
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Checkpoint Found!!!!")
        model.load_weights(latest_checkpoint)
    else:
        print("No model found, Everything will start from epoch 0")
    model.fit(X_train, y_train, epochs=MAX_EPOCHS, batch_size=BATCH, verbose=1,
              callbacks=[LearningRateReducerCb(), tensorboard_callback, cp_callback],
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
