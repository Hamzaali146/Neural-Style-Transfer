#!pip install tensorflow opencv-python matplotlib numpy

# from opencv we are using PIL package which is Pillow from this package we are using basic funcions like load
# and preprocess image

import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import cv2

#Keras is basically a package in which most popular AI\ML algo are coded and implemented! we are just using and pretraining it

def load_and_process_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    # print(img)
    return img

content_image = load_and_process_image('/content/content.jpeg', (224, 224))
style_image = load_and_process_image('/content/style.jpg', (224, 224))


def get_model():
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
    content_layer = 'block5_conv1'


    output_layers = []
    for name in style_layers:
        output_layers.append(vgg.get_layer(name).output)


    output_layers.append(vgg.get_layer(content_layer).output)

    model = Model(inputs=vgg.input, outputs=output_layers)
    return model

model = get_model()


content_weight = 1  # this is 1 because our targeted generated image is same as content image at initial there will be no content loss at start gradually will gonna increase.
style_weight = 10000

def content_loss(content, generated):
    # print(content.shape)
    # print(generated.shape)
    return (tf.reduce_mean(tf.square(content - generated)))/2

def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    matrix = tf.reshape(tensor, [-1, channels])
    return tf.matmul(matrix, matrix, transpose_a=True)

def style_loss(style, generated):
    S = gram_matrix(style)
    G = gram_matrix(generated)
    return tf.reduce_mean(tf.square(S - G))

def total_loss(outputs, content_targets, style_targets):
    style_outputs, content_outputs = outputs[:4], outputs[4]
    content_loss_value = content_loss(content_targets, content_outputs)
    style_loss_value = tf.add_n([style_loss(style_targets[i], style_outputs[i]) for i in range(len(style_outputs))])
    print(len(style_outputs))
    total_loss_value = (content_loss_value * content_weight) + (style_loss_value * style_weight)
    return total_loss_value


def compute_loss_and_grads(model, generated_image, content_targets, style_targets):
    with tf.GradientTape() as tape:
        outputs = model(generated_image)
        loss_value = total_loss(outputs, content_targets, style_targets)
    grads = tape.gradient(loss_value, generated_image)
    return loss_value, grads

def style_transfer(model, content_image, style_image, iterations=5000):
    generated_image = tf.Variable(content_image, dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate=0.02)
    content_targets = model(content_image)[4]  # Content layer
    style_targets = model(style_image)[:4]     # Style layers

    for i in range(iterations):
        loss_value, grads = compute_loss_and_grads(model, generated_image, content_targets, style_targets)
        optimizer.apply_gradients([(grads, generated_image)])
        if i % 100 == 0:
            print(f"Iteration: {i}, Loss: {loss_value}")
    print("NTS production by theboys👻")
    return generated_image

def deprocess_image(img):
    img = img[0]
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]  # BGR to RGB
    img = np.clip(img, 0, 255).astype('uint8')
    return img

generated_image = style_transfer(model, content_image, style_image)
plt.imshow(deprocess_image(generated_image.numpy()))
plt.show()
