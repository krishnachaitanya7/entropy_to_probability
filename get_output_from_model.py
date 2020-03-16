import tensorflow as tf


class entropy_to_probability:
    def __init__(self, model_path):
        """
        This is init class for the file. You have to instantiate the class and use it in your scripts
        :param model_path: str
            Pass the full path of the HDF5 model
        """
        self.path_to_h5 = model_path
        self.my_model = tf.keras.models.load_model(self.path_to_h5)

    def predict_probability(self, entropy: float):
        """
        Input each float entropy value and get the predicted output
        :param entropy: float
            Pass a single value for the model to predict
        :return: float
            The predicted probability of corresponding entropy
        """
        return self.my_model.predict([[[entropy]]])


if __name__ == "__main__":
    # my_model is the folder name which contains the saved model (saved_model.pb and variables folder)
    # In yiour case pass the full path, or if in same directory you can just pass the folder name
    etp = entropy_to_probability("my_model")
    # Then you can pass each entropy value like below
    # 0.3098884946900000159 is the entropy
    # pred_prob is the predicted probability
    pred_prob = etp.predict_probability(0.3098884946900000159)
    print(pred_prob)
