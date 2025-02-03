# Blurry Image Detection using FFT (Fast Fourier Transform)

This is follow-up for the Capstone 1 project, see [Blurry-Image-Detection](https://github.com/alex-s-888/Blurry-Image-Detection)  
In the first project we built CNN model that predicts quality of images - if image is blurred or not. Deep Neural Networks are usually costly as they require lots of resources and time.
The goal of this second project is to try alternative (and cheaper) approach to detect blurriness, namely `Fast Fourier Transform`, which is available out-of-the-box in several python libraries, e.g. `numpy` 


## Project structure

### model
Contains the source code  
- [notebook.ipynb](model/notebook.ipynb) - notebook to do research of FFT and define functions to be used in predictions.
- [train.py](model/train.py) - python script that builds CNN model and serializes it as `.tflite` file.

### deployment_local
Contains `app.py` script to launch [Gradio](https://www.gradio.app/) application that performs blur detection using both CNN and FFT, so the results may be compared.

The application (same `app.py` as mentioned above) is also deployed to Hugging Face cloud platform. The link is  
[https://huggingface.co/spaces/Alex-MMXXIV/BlurDetection](https://huggingface.co/spaces/Alex-MMXXIV/BlurDetection)

