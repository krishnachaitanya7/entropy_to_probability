import tensorflow as tf

class entropy_to_probability:
    def __init__(self, path_to_h5):
        """
        This is init class for the file. You have to instantiate the class and use it in your scripts
        :param path_to_h5: str
            Pass the full path of the HDF5 model
        """
        self.path_to_h5 = path_to_h5
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
    etp = entropy_to_probability("my_model/entropy_to_probability.h5")
    pred_prob = etp.predict_probability(0.00274605262826)
    print(pred_prob)

