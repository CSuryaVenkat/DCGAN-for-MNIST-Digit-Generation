<h1>DCGAN for MNIST Digit Generation</h1>

<p>This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) using TensorFlow and Keras to generate handwritten digit images similar to those in the MNIST dataset.</p>


<h2>Components</h2>

<h3>Hyperparameters</h3>

<ul>
  <li><strong>DROPOUT:</strong> Dropout rate to prevent overfitting by randomly setting a fraction of input units to 0 during training.</li>
  <li><strong>WINDOW_SIZE:</strong> The size of the input sequence, derived from SEQ_LEN.</li>
</ul>

<h3>Model Architecture</h3>

<ul>
  <li><strong>Discriminator:</strong> A convolutional neural network (CNN) that classifies images as real or fake.</li>
  <li><strong>Generator:</strong> A deconvolutional neural network that generates fake images from random noise.</li>
</ul>

<h2>Key Functions</h2>

<ul>
  <li><strong>plot_images:</strong> Function to plot images in a grid format.</li>
  <li><strong>GenerateSamplesCallback:</strong> A callback to generate and save images from the generator at the end of each epoch.</li>
  <li><strong>build_discriminator:</strong> Constructs the discriminator model.</li>
  <li><strong>build_generator:</strong> Constructs the generator model.</li>
  <li><strong>DCGAN:</strong> Custom model class for the DCGAN that defines the training step.</li>
  <li><strong>train_dcgan_mnist:</strong> Function to train the DCGAN on the MNIST dataset.</li>
</ul>

<h2>Implementation Steps</h2>

<ol>
  <li><strong>Prepare Data:</strong>
    <p>The MNIST dataset is automatically downloaded and preprocessed to scale the images between [-1, 1].</p>
  </li>
  <li><strong>Build and Compile Models:</strong>
    <p>The discriminator and generator models are built using convolutional and deconvolutional layers respectively. The DCGAN model is compiled with Adam optimizers and binary cross-entropy loss.</p>
  </li>
  <li><strong>Train the Model:</strong>
    <p>The DCGAN is trained for 50 epochs with a batch size of 32. A callback generates and saves sample images at the end of each epoch.</p>
  </li>
  <li><strong>Generate Images:</strong>
    <p>After training, the generator can be used to create new handwritten digit images.</p>
  </li>
</ol>

<h2>Output</h2>

<h3>Training Loss</h3>

<p>The training loss for both the generator and the discriminator is logged and can be plotted to monitor training progress.</p>

<h3>Generated Images</h3>

<p>Sample images generated by the model are saved in the <code>g1</code> directory at the end of each epoch.</p>



