import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from myfuns.funs import draw_prediction_on_image

MODEL_URLS = {
    "movenet_lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
    "movenet_thunder": "https://tfhub.dev/google/movenet/singlepose/thunder/4"
}

def load_model(model_name):
    if model_name not in MODEL_URLS:
        raise ValueError(f"Unsupported model name: {model_name}")
    module = hub.load(MODEL_URLS[model_name])
    input_size = 192 if "lightning" in model_name else 256
    def movenet(input_image):
        model = module.signatures['serving_default']
        input_image = tf.cast(input_image, dtype=tf.int32)
        outputs = model(input_image)
        return outputs['output_0'].numpy()
    return movenet, input_size

def preprocess_image(image_path, input_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize_with_pad(tf.expand_dims(image, 0), input_size, input_size)
    return image

def main():
    model_name = "movenet_lightning"
    image_path = "input_image.jpeg"
    try:
        movenet, input_size = load_model(model_name)
        input_image = preprocess_image(image_path, input_size)
        keypoints_with_scores = movenet(input_image)
        display_image = tf.image.resize_with_pad(tf.expand_dims(tf.image.decode_jpeg(tf.io.read_file(image_path)), 0), 1280, 1280)
        output_overlay = draw_prediction_on_image(np.squeeze(display_image.numpy(), 0), keypoints_with_scores)
        plt.figure(figsize=(5, 5))
        plt.imshow(output_overlay)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()