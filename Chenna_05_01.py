

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_images(g1, n_rows=1, n_cols=10):
    """
    Plot the images in a 1x10 grid
    :param g1:
    :return:
    """
    f, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    ax = ax.flatten()
    for i in range(n_rows*n_cols):
        ax[i].imshow(g1[i, :], cmap='gray')
        ax[i].axis('off')
    return f, ax

class GenerateSamplesCallback(tf.keras.callbacks.Callback):
    """
    Callback to generate images from the generator network at the end of each epoch
    Uses the same noise vector to generate images at each epoch, so that the images can be compared over time
    """
    def __init__(self, generator, noise):
        self.generator = generator
        self.noise = noise

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists("g1"):
            os.mkdir("g1")
        g1 = self.generator(self.noise, training=False)
        g1 = g1.numpy()
        g1 = g1*127.5 + 127.5
        g1 = g1.reshape((10, 28, 28))
        # plot images using matplotlib
        plot_images(g1)
        plt.savefig(os.path.join("g1", f"g1_{epoch}.png"))
        # close the plot to free up memory
        plt.close()

def build_discriminator():
    network = tf.keras.models.Sequential()
    conv2dLayer1=tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1))
    network.add(conv2dLayer1)
    leakyRelu=tf.keras.layers.LeakyReLU()
    network.add(leakyRelu)
    dropout=tf.keras.layers.Dropout(0.3)
    network.add(dropout)
    network.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    network.add(tf.keras.layers.LeakyReLU())
    Relu1=tf.keras.layers.Dropout(0.3)
    network.add(Relu1)
    network.add(tf.keras.layers.Flatten())
    network.add(tf.keras.layers.Dense(1))

    return network

def build_generator():
    
    network = tf.keras.models.Sequential()
    dense1=tf.keras.layers.Dense(7*7*8, use_bias=False, input_shape=(100,))
    network.add(dense1)
    network.add(tf.keras.layers.BatchNormalization())
    Relu3=tf.keras.layers.LeakyReLU()
    network.add(Relu3)
    Reshape_layer=tf.keras.layers.Reshape((7, 7, 8))
    network.add(Reshape_layer)
    conv2DT=tf.keras.layers.Conv2DTranspose(8, kernel_size=(5,5), strides=(1,1), padding='same', use_bias=False)
    network.add(conv2DT)
    Batch=tf.keras.layers.BatchNormalization()
    network.add(Batch)
    network.add(tf.keras.layers.LeakyReLU())
    conv2DT1=tf.keras.layers.Conv2DTranspose(16, kernel_size=(5,5), strides=(2,2), padding='same', use_bias=False)
    network.add(conv2DT1)
    network.add(tf.keras.layers.BatchNormalization())
    network.add(tf.keras.layers.LeakyReLU())
    network.add(tf.keras.layers.Conv2DTranspose(1, kernel_size=(5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))


    return network

class DCGAN(tf.keras.Model):
    def __init__(self, discriminator, generator):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = 100

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        bs = tf.shape(data)[0]

        # generate random noise for the generator input
        noise = tf.random.uniform([bs, self.noise_dim])

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
        # generate fake images using the generator network
            g1 = self.generator(noise, training=True)

            # pass real and fake images through discriminator
            r1 = self.discriminator(data, training=True)
            f1 = self.discriminator(g1, training=True)

            # calculate losses for discriminator
            L1 = self.loss_fn(tf.ones_like(r1), r1)
            L2 = self.loss_fn(tf.zeros_like(f1), f1)
            d_loss = L1*0.5 + L2*0.5

            # calculate loss for generator
            g_loss = self.loss_fn(tf.ones_like(f1), f1)

        # calculate gradients and update weights for both generator and discriminator
        gradients_of_generator = tape1.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = tape2.gradient(d_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}


def train_dcgan_mnist():
    tf.keras.utils.set_random_seed(5368)
    # load mnist
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # the images are in the range [0, 255], we need to rescale them to [-1, 1]
    x_train = (x_train - 127.5) / 127.5
    x_train = x_train[..., tf.newaxis].astype(np.float32)

    # plot 10 random images
    example_images = x_train[:10]*127.5 + 127.5
    plot_images(example_images)

    plt.savefig("real_images.png")


    # build the discriminator and the generator
    d = build_discriminator()
    g = build_generator()


    # build the DCGAN
    dcgan = DCGAN(discriminator=d, generator=g)

    # compile the DCGAN
    dcgan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  g_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    callbacks = [GenerateSamplesCallback(generator, tf.random.uniform([10, 100]))]
    # train the DCGAN
    dcgan.fit(x_train, epochs=50, bs=32, callbacks=callbacks, shuffle=True)

    # generate images
    noise = tf.random.uniform([16, 100])
    g1 = generator(noise, training=False).numpy().tolist()
    plot_images(g1*127.5 + 127.5, 4, 4)
    plt.savefig("g1.png")

    generator.save('generator.h5')


if __name__ == "__main__":
    train_dcgan_mnist()
