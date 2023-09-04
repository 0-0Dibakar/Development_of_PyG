import torch
import tensorflow as tf

class FrameworkHandler:
    def __init__(self, framework):
        self.framework = framework
        self.model = None

    def load_model(self, model_path):
        if self.framework == "pytorch":
            self.model = torch.load(model_path)
        elif self.framework == "tensorflow":
            self.model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError("Unsupported framework")

    def predict(self, input_data):
        if self.model is None:
            raise ValueError("Model has not been loaded")

        if self.framework == "pytorch":
            input_tensor = torch.from_numpy(input_data)
            with torch.no_grad():
                output = self.model(input_tensor)
            return output.numpy()
        elif self.framework == "tensorflow":
            input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
            output = self.model(input_tensor)
            return output.numpy()
        else:
            raise ValueError("Unsupported framework")

if __name__ == "__main__":
    # Example usage
    pytorch_handler = FrameworkHandler("pytorch")
    pytorch_handler.load_model("pytorch_model.pth")
    pytorch_input = torch.rand(1, 3, 224, 224)  # Example input data
    pytorch_output = pytorch_handler.predict(pytorch_input.numpy())
    print("PyTorch output:", pytorch_output)

    tensorflow_handler = FrameworkHandler("tensorflow")
    tensorflow_handler.load_model("tensorflow_model.h5")
    tensorflow_input = tf.random.uniform((1, 224, 224, 3))  # Example input data
    tensorflow_output = tensorflow_handler.predict(tensorflow_input.numpy())
    print("TensorFlow output:", tensorflow_output)
