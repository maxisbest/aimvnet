import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from myfuns.funs import draw_prediction_on_image, draw_prediction_on_image2

def load_model(model_name):
    urls = {
        "movenet_lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
        "movenet_thunder": "https://tfhub.dev/google/movenet/singlepose/thunder/4"
    }
    if model_name not in urls:
        raise ValueError(f"Unsupported model name: {model_name}")
    module = hub.load(urls[model_name])
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
    movenet, input_size = load_model(model_name)
    input_image = preprocess_image(image_path, input_size)
    keypoints_with_scores = movenet(input_image)
    # 读取原始图片用于可视化
    orig_image = tf.io.read_file(image_path)
    orig_image = tf.image.decode_jpeg(orig_image)
    orig_image = tf.image.resize_with_pad(tf.expand_dims(orig_image, 0), 1280, 1280)
    orig_image_np = np.squeeze(orig_image.numpy(), 0)
    # 关键：将关键点和骨架叠加到原始图片上
    output_overlay = draw_prediction_on_image2(orig_image_np, keypoints_with_scores)
    plt.figure(figsize=(5, 5))
    plt.imshow(output_overlay)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()