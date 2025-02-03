import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import datetime


class TFLiteModel:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, *data_args):
        assert len(data_args) == len(self.input_details)
        for data, details in zip(data_args, self.input_details):
            self.interpreter.set_tensor(details["index"], data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])


model = TFLiteModel("../model/mymodel.tflite")
mylabels = ["Blurred", "Not Blurred"]


def to_grayscale(image_array):
  return np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])


def is_blurred(image_array, blur_threshold = 133):
  """
  Detects if an image is blurred using the FFT.
  """

  # Perform FFT
  f = np.fft.fft2(image_array)
  fshift = np.fft.fftshift(f)
  magnitude_spectrum = 20 * np.log(np.abs(fshift))

  # Calculate the mean of the magnitude spectrum
  mean_magnitude = np.mean(magnitude_spectrum)

  # Return True if mean magnitude is below the threshold (more blur)
  return mean_magnitude, mean_magnitude < blur_threshold


def predict_blur(input_image):

    print(datetime.datetime.now())

    pilImage = Image.fromarray(input_image)
    if pilImage.mode != 'RGB':
        pilImage = pilImage.convert(mode='RGB')
    resized_image = pilImage.resize((256, 256))

    arr_image = np.asarray(resized_image)
    img = np.float32(arr_image/255)

    label = mylabels[ model.predict([img])[0].argmax() ]

    mean_magnitude, is_image_blurred = is_blurred(to_grayscale(input_image))
    labelFFT = mylabels[0] if is_image_blurred else mylabels[1]

    return "Prediction using CNN: " + label, "Prediction using FFT: " + labelFFT


demo = gr.Interface(
    fn = predict_blur,
    inputs = ["image"],
    outputs = ["text", "text"],
)

demo.launch()