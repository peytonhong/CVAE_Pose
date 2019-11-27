from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
import cv2
import plot_utils

from IPython import display

""" Generate Rectangles """
def get_points(rec_size, center, theta):
        points = np.matrix([
                            [-rec_size[0]/2, -rec_size[1]/2],
                            [-rec_size[0]/2, +rec_size[1]/2],
                            [+rec_size[0]/2, +rec_size[1]/2],
                            [+rec_size[0]/2, -rec_size[1]/2]])
        points = np.vstack((points.transpose(), np.ones((1,4))))
        theta = theta*np.pi/180
        R = np.matrix([[np.cos(theta), -np.sin(theta)], 
                      [np.sin(theta),  np.cos(theta)]])
        
        T = np.hstack((R, center))
        T = np.vstack((T, np.array([0, 0, 1])))
        points = T*points
        return points[:2].transpose().astype(np.int32)

def get_rectangles(NUM_IMAGES, HEIGHT, WIDTH):
    imgs = []
    rec_size = (30,30)
    for x, y, theta in zip(np.random.rand(NUM_IMAGES)*WIDTH, np.random.rand(NUM_IMAGES)*HEIGHT, np.random.rand(NUM_IMAGES)*90):
        # center = np.array([[x], [y]])
        center = np.array([[56], [56]])
        img = np.zeros((HEIGHT, WIDTH, 1), dtype=np.uint8)
        cv2.fillPoly(img, [get_points(rec_size, center, theta)], color=(255,255,255))
        imgs.append(img)
    return np.array(imgs)

def generate_dataset(BATCH_SIZE):
    HEIGHT, WIDTH = 112, 112
    images = get_rectangles(BATCH_SIZE, HEIGHT, WIDTH)
    images = images.astype(np.float32) / 255.
    images[images >= .5] = 1.
    images[images < .5] = 0.
    # dataset = tf.data.Dataset.from_tensor_slices(images).batch(BATCH_SIZE)    
    
    # return images.reshape((-1,HEIGHT*WIDTH))
    return images


# (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

# # Normalizing the images to the range of [0., 1.]
# train_images /= 255.
# test_images /= 255.

# # Binarization
# train_images[train_images >= .5] = 1.
# train_images[train_images < .5] = 0.
# test_images[test_images >= .5] = 1.
# test_images[test_images < .5] = 0.

# TRAIN_BUF = 60000
# BATCH_SIZE = 100

# TEST_BUF = 10000

# train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
# test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)
test_dataset = generate_dataset(100)

class CVAE(tf.keras.Model):
  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.inference_net = tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=(112, 112, 1)),
          tf.keras.layers.Conv2D(
              filters=16, kernel_size=3, strides=(2, 2), activation='relu'),
          tf.keras.layers.Conv2D(
              filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
          tf.keras.layers.Conv2D(
              filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
          tf.keras.layers.Flatten(),
          # No activation
          tf.keras.layers.Dense(latent_dim + latent_dim),
      ]
    )

    self.generative_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
          tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
          tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
          tf.keras.layers.Conv2DTranspose(
              filters=64,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
          tf.keras.layers.Conv2DTranspose(
              filters=32,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
          tf.keras.layers.Conv2DTranspose(
              filters=16,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
          # No activation
          tf.keras.layers.Conv2DTranspose(
              filters=1, kernel_size=3, strides=(2, 2), padding="SAME"),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.generative_net(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs

    return logits

optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

@tf.function
def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)

  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def compute_apply_gradients(model, x, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

epochs = 100
latent_dim = 2
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

def generate_and_save_images(model, epoch, test_input):
  predictions = model.sample(test_input)
  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0], cmap='gray')
      plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  # plt.show()

generate_and_save_images(model, 0, random_vector_for_generation)

# RESULTS_DIR
RESULTS_DIR = 'results'
try:
    os.mkdir(RESULTS_DIR)
except(FileExistsError):
    pass
# delete all existing files
files = glob.glob(RESULTS_DIR+'/*')
for f in files:
    os.remove(f)

# Plot for manifold learning result
PMLR = plot_utils.Plot_Manifold_Learning_Result(RESULTS_DIR, 20, 20, 112, 112, 1.0, 2.0)

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for _ in range(1000):
    train_x = generate_dataset(100)
    compute_apply_gradients(model, train_x, optimizer)
  end_time = time.time()

  if epoch % 1 == 0:
    loss = tf.keras.metrics.Mean()
    
    loss(compute_loss(model, test_dataset))
    elbo = -loss.result()
    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, '
          'time elapse for current epoch {}'.format(epoch,
                                                    elbo,
                                                    end_time - start_time))
    generate_and_save_images(
        model, epoch, random_vector_for_generation)
    # Plot for manifold learning result
    if latent_dim == 2:       
        y_PMLR = model.sample(PMLR.z)
        y_PMLR_img = tf.reshape(y_PMLR, [PMLR.n_tot_imgs, 112, 112])
        PMLR.save_images(y_PMLR_img.numpy(), name="/PMLR_epoch_%02d" % (epoch) + ".jpg")

def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

plt.imshow(display_image(epochs))
plt.axis('off')# Display images

anim_file = 'cvae.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

import IPython
if IPython.version_info >= (6,2,0,''):
  display.Image(filename=anim_file)

