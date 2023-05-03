import argparse
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
import pprint
# %matplotlib inline

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import soundfile as sf
import cv2


WTS = ['/kaggle/input/vgg-finetuned-weights/vgg19_fine_tuned_window_size.h5', '/kaggle/input/vgg-finetuned-weights/vgg19_fine_tuned_original_18.h5', '/kaggle/input/vgg-finetuned-weights/vgg19_fine_tuned_spec_augment.h5', '/kaggle/input/vgg-finetuned-weights/vgg19_fine_tuned_mel.h5', '/kaggle/input/vgg-finetuned-weights/vgg19_fine_tuned.h5', None]
CONTENT_WEIGHTS = [1e-2]
STYLE_WEIGHTS = [1e4]
DTWS = [True, False]
epochs=20000
lr = 1e-3

for model_weights in WTS:
    print('model_weights: ', model_weights)
    for i in range(6):
        content_weight = CONTENT_WEIGHTS[i]
        print('content_weight:', content_weight)
        style_weight = STYLE_WEIGHTS[i]
        print('style_weight:', style_weight)
        
        for j in range(2):
            dtw = DTWS[j]
            if dtw:
                print("Dynamic Time Warping used............")

            # Load the WAV file - content
            filename = '/kaggle/input/indian-american/american_content.wav'
            if dtw:
                filename = '/kaggle/input/indian-american/american_content_warped.wav'

            y, sr = librosa.load(filename, sr=None)

            # Create the spectrogram
            spectrogram = librosa.stft(y)
            spectrogram_phase = np.angle(spectrogram)
            spectrogram_db = librosa.amplitude_to_db(abs(spectrogram)) # spectrogram_db is in log scale
            cv2.imwrite('american.png', spectrogram_db)

            # sr = 22050
            img = cv2.imread('american.png', cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            # print(img.shape)
            # gray_channel = np.mean(img, axis=2)

            # Combine magnitude and phase components
            # S_combined = librosa.db_to_amplitude(img) * np.exp(1j * spectrogram_phase)
            y_inv = librosa.griffinlim(librosa.db_to_amplitude(img)) # reconstruction of audio signal from grayscale image
            # y_inv = librosa.griffinlim(S_combined) # Integrating phase is slightly blurring the audio
            # plt.imshow(gray_channel, cmap='gray', origin='lower')

            sf.write('output_american.wav', y_inv, sr)

            # Load the WAV file
            filename = '/kaggle/input/indian-american/indian_style.wav'
            if dtw:
                filename = '/kaggle/input/indian-american/indian_style_warped.wav'

            y, sr = librosa.load(filename, sr=16000) # passing in the sr of american

            # Create the spectrogram
            spectrogram = librosa.stft(y)
            spectrogram_phase = np.angle(spectrogram)
            spectrogram_db = librosa.amplitude_to_db(abs(spectrogram)) # spectrogram_db is in log scale
            cv2.imwrite('indian.png', spectrogram_db)

            img = cv2.imread('indian.png', cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            # print(img.shape)
            # gray_channel = np.mean(img, axis=2)
            # Combine magnitude and phase components
            # S_combined = librosa.db_to_amplitude(img) * np.exp(1j * spectrogram_phase)
            y_inv = librosa.griffinlim(librosa.db_to_amplitude(img)) # reconstruction of audio signal from grayscale image
            # y_inv = librosa.griffinlim(S_combined) # Integrating phase is slightly blurring the audio
            # plt.imshow(gray_channel, cmap='gray', origin='lower')

            sf.write('output_indian.wav', y_inv, sr)

            train_dir = '/kaggle/input/speech-accent-archive/clips/train/'
            validation_dir = '/kaggle/input/speech-accent-archive/clips/validation/'
            BATCH_SIZE = 32
            IMG_SIZE = (1025, 216) # taken from the tutorial (this can effect the training weights)

            train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
            validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

            print(validation_dataset)
            NUM_CLASSES = 18
            if model_weights == WTS[-2]:
                NUM_CLASSES=200

            from tensorflow.keras.applications.vgg19 import preprocess_input # pre-process input to scale images

            tf.random.set_seed(272) # DO NOT CHANGE THIS VALUE
            pp = pprint.PrettyPrinter(indent=4)
            vgg_func = tf.keras.applications.VGG19(include_top=False, input_shape=(1025, 216, 3), weights='imagenet')
            if model_weights is not None:
                vgg = tf.keras.Sequential()
                vgg.add(vgg_func)
                vgg.add(tf.keras.layers.GlobalAveragePooling2D())
                vgg.add(tf.keras.layers.Dropout(0.2))
                vgg.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
                vgg.load_weights(model_weights)
                vgg = vgg.layers[0]
                if model_weights == '/kaggle/input/vgg-finetuned-weights/vgg19_fine_tuned.h5':
                    model_weights_1='vgg19_fine_tuned'
                elif model_weights == '/kaggle/input/vgg-finetuned-weights/vgg19_fine_tuned_mel.h5':
                    model_weights_1='vgg19_fine_tuned_mel'
                elif model_weights == '/kaggle/input/vgg-finetuned-weights/vgg19_fine_tuned_spec_augment.h5':
                    model_weights_1='vgg19_fine_tuned_spec_augment'
                elif model_weights == '/kaggle/input/vgg-finetuned-weights/vgg19_fine_tuned_original_18.h5':
                    model_weights_1='vgg19_fine_tuned_original_18'
                elif model_weights == '/kaggle/input/vgg-finetuned-weights/vgg19_fine_tuned_window_size.h5':
                    model_weights_1='vgg19_fine_tuned_window_size'
            else:
                model_weights_1='imagenet'
                vgg = vgg_func

            vgg.trainable = False
            pp.pprint(vgg)
            vgg.summary()

            content_image = cv2.imread('/kaggle/working/american.png').astype(np.uint8)
            content_image.shape

            def compute_content_cost(content_output, generated_output):
                """
                Computes the content cost

                Arguments:
                a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
                a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

                Returns: 
                J_content -- the content cost
                """

                a_C = content_output[-1]
                a_G = generated_output[-1]

                # Retrieve dimensions from a_G
                _, n_H, n_W, n_C = a_G.get_shape().as_list()

                # Reshape a_C and a_G
                a_C_unrolled = tf.reshape(a_C, shape=[_, n_H * n_W, n_C])
                a_G_unrolled = tf.reshape(a_G, shape=[_, n_H * n_W, n_C])

                # compute the cost with tensorflow
                J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)), axis=None) / (4 * n_H * n_W * n_C)  # axis = None means sum up all the elements in the tensor

                return J_content

            example = cv2.imread('/kaggle/working/indian.png').astype(np.uint8)
            example.shape

            def gram_matrix(A):
                """
                Argument:
                A -- matrix of shape (n_C, n_H*n_W)

                Returns:
                GA -- Gram matrix of A, of shape (n_C, n_C)
                """  
                GA = tf.linalg.matmul(A, tf.transpose(A))   # tf.transpose(A) = tf.transpose(A, [1, 0])
                return GA

            def compute_layer_style_cost(a_S, a_G):
                """
                Arguments:
                a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
                a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

                Returns: 
                J_style_layer -- tensor representing a scalar value, style cost
                """
                # Retrieve dimensions from a_G
                _, n_H, n_W, n_C = a_G.get_shape().as_list()

                # Reshape the images from (n_H * n_W, n_C) to have them of shape (n_C, n_H * n_W)
                a_S = tf.transpose(tf.reshape(a_S, shape=[n_H * n_W, n_C]))
                a_G = tf.transpose(tf.reshape(a_G, shape=[n_H * n_W, n_C]))

                # Computing gram_matrices for both images S and G
                GS = gram_matrix(a_S)
                GG = gram_matrix(a_G)

                # Computing the loss
                J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG)), axis=None) / (4.0 * ((n_H*n_W*n_C)**2))
                return J_style_layer

            
            for layer in vgg.layers:
                print(layer.name)
                
            vgg.get_layer('block5_conv4').output


            STYLE_LAYERS = [
                ('block1_conv1', 0.2),
                ('block2_conv1', 0.2),
                ('block3_conv1', 0.2),
                ('block4_conv1', 0.2),
                ('block5_conv1', 0.2)]

            def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
                """
                Computes the overall style cost from several chosen layers

                Arguments:
                style_image_output -- our tensorflow model
                generated_image_output --
                STYLE_LAYERS -- A python list containing:
                                    - the names of the layers we would like to extract style from
                                    - a coefficient for each of them

                Returns: 
                J_style -- tensor representing a scalar value, style cost
                """

                # initialize the overall style cost
                J_style = 0

                # Set a_S to be the hidden layer activation from the layer we have selected.
                # The last element of the array contains the content layer image, which must not be used.
                a_S = style_image_output[:-1]

                # Set a_G to be the output of the choosen hidden layers.
                # The last element of the list contains the content layer image which must not be used.
                a_G = generated_image_output[:-1]
                for i, weight in zip(range(len(a_S)), STYLE_LAYERS):  
                    # Compute style_cost for the current layer
                    J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

                    # Add weight * J_style_layer of this layer to overall style cost
                    J_style += weight[1] * J_style_layer

                return J_style

            @tf.function()
            def total_cost(J_content, J_style, alpha = 10, beta = 40):
                """
                Computes the total cost function

                Arguments:
                J_content -- content cost coded above
                J_style -- style cost coded above
                alpha -- hyperparameter weighting the importance of the content cost
                beta -- hyperparameter weighting the importance of the style cost

                Returns:
                J -- total cost
                """
                J = alpha * J_content + beta * J_style
                return J

            content_image = cv2.imread('american.png').astype(np.uint8)
            content_image = cv2.resize(content_image, (216, 1025)).astype(np.uint8)
            content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))
            content_image = preprocess_input(content_image)
            imshow(content_image[0])
            plt.show()

            style_image = cv2.imread('indian.png').astype(np.uint8)
            style_image = cv2.resize(style_image, (216, 1025)).astype(np.uint8)
            style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))
            imshow(style_image[0])
            plt.show()

            generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
            generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
            imshow(generated_image.numpy()[0])
            plt.show()

            def get_layer_outputs(vgg, layer_names):
                """ Creates a vgg model that returns a list of intermediate output values."""
                outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

                model = tf.keras.Model([vgg.input], outputs)
                return model

            content_layer = [('block5_conv4', 1)]

            vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)
            content_target = vgg_model_outputs(content_image)  # Content encoder
            style_targets = vgg_model_outputs(style_image)     # Style enconder

            preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
            print(preprocessed_content.shape)
            a_C = vgg_model_outputs(preprocessed_content) ## computes activations of the style layers + content layer for content image

            preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
            a_S = vgg_model_outputs(preprocessed_style)  ## computes activations of the style layers + content layer for style image

            def clip_0_1(image):
                """
                Truncate all the pixels in the tensor to be between 0 and 1

                Arguments:
                image -- Tensor
                J_style -- style cost coded above

                Returns:
                Tensor
                """
                return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

            def tensor_to_image(tensor):
                """
                Converts the given tensor into a PIL image

                Arguments:
                tensor -- Tensor

                Returns:
                Image: A PIL image
                """
                tensor = tensor * 255
                tensor = np.array(tensor, dtype=np.uint8)
                if np.ndim(tensor) > 3:
                    assert tensor.shape[0] == 1
                    tensor = tensor[0]
                return Image.fromarray(tensor)

            generated_image = tf.Variable(generated_image)

            optimizer = tf.keras.optimizers.Adam(learning_rate=lr) 

            @tf.function()
            def train_step(generated_image):
                with tf.GradientTape() as tape:
                    # In this function we use the precomputed encoded images a_S and a_C
                    # Compute a_G as the vgg_model_outputs for the current generated image

                    a_G = vgg_model_outputs(generated_image)

                    # Compute the style cost
                    J_style = compute_style_cost(a_S, a_G)
                    # Compute the content cost
                    J_content = compute_content_cost(a_C, a_G)
                    # Compute the total cost
                    J = total_cost(J_content, J_style, alpha=content_weight, beta=style_weight)

                grad = tape.gradient(J, generated_image)

                optimizer.apply_gradients([(grad, generated_image)])
                generated_image.assign(clip_0_1(generated_image))
                return J_content, J_style, J

            epochs = epochs
            for i in range(epochs):
                J_content, J_style, J = train_step(generated_image)
                if i % 1000 == 0:
                    print(f"J_content: {J_content}, J_style: {J_style}, J_total: {J}")
                if i % 1000 == 0:
                    print(f"Epoch {i} ")
                if i % 1000 == 0:
                    image = tensor_to_image(generated_image)
                    image.save(f"image_{i}.png")

            image = tensor_to_image(generated_image)
            image.save(f'generated_{content_weight}_{style_weight}_{dtw}_{model_weights_1}.png')
