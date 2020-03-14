import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
import numpy as np

changed_lr = False


class LearningRateReducerCb(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        global changed_lr
        if not changed_lr and logs['mse'] < 2e-05:
            old_lr = self.model.optimizer.lr.read_value()
            new_lr = old_lr / 10
            print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
            self.model.optimizer.lr.assign(new_lr)
            changed_lr = True


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
    model = keras.Sequential()
    model.add(layers.SimpleRNN(4, return_sequences=True))
    model.add(layers.SimpleRNN(4))
    model.add(layers.Dense(1, activation=tf.keras.activations.sigmoid))
    model.compile(loss=tf.keras.losses.mean_squared_error,
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['mse'])
    model.fit(X_train, y_train, epochs=MAX_EPOCHS, batch_size=BATCH, verbose=1, callbacks=[LearningRateReducerCb()],
              validation_data=(X_test, y_test))
    y_pred = model.predict(X_val)
    rmse_calculated = mean_squared_error(y_pred.flatten(), y_val.flatten())
    print(f"RMSE on Validation: {rmse_calculated}")
    final_save_matrix = np.stack((X_val.flatten(), y_val.flatten(), y_pred.flatten()), axis=1)
    open(file_name, 'w').close()
    np.savetxt(file_name, final_save_matrix,
               delimiter=',', header="X_validation,Original_Y,Predicted_y", comments="")
    # Noted RMSE: 9.070411888314994e-06