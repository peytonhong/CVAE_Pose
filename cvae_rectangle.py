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
import argparse
from cvae import *
from IPython import display
from rectangles import get_points, get_rectangles, generate_dataset

"""parsing and configuration"""
def argparse_args():  
  desc = "Tensorflow 2.0 implementation of 'Convolutional Augmented Variational AutoEncoder (CAVAE)'"
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('command', help="'train' or 'evaluate'")
  parser.add_argument('--latent_dim', default=2, type=int, help="Dimension of latent vector")
  parser.add_argument('--num_epochs', default=200, type=int, help="The number of epochs to run")

  return parser.parse_args()

"""main function"""
def main(args):  

  test_x, test_y, test_pose = generate_dataset(100)

  epochs = args.num_epochs
  latent_dim = args.latent_dim
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
    plt.savefig('./results/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()
    plt.close(fig)

  # generate_and_save_images(model, 0, random_vector_for_generation)

  if args.command == 'train':

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
    PMLR = plot_utils.Plot_Manifold_Learning_Result(RESULTS_DIR, 20, 20, 112, 112, 1.0, 5.0)

    loss_list = []
    elbo_prev = -np.inf

    optimizer = tf.keras.optimizers.Adam(1e-4)
    
    for epoch in range(1, epochs + 1):
      start_time = time.time()
      for _ in range(1000):
        train_x, train_y, train_pose = generate_dataset(100)
        compute_apply_gradients(model, train_x, train_y, train_pose, optimizer)
      end_time = time.time()

      if epoch % 1 == 0:
        loss = tf.keras.metrics.Mean()
        
        loss(compute_loss(model, test_x, test_y, test_pose))
        elbo = -loss.result()
        display.clear_output(wait=False)
        
        loss_list.append([epoch, elbo])
        
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

        # save checkpoint
        if elbo > elbo_prev:
          model.save_weights('./checkpoints/cvae_rectangle_checkpoint')
          elbo_prev = elbo

        # save loss curve
        plt.plot(np.array(loss_list)[:,0], np.array(loss_list)[:,1])
        plt.savefig('./results/loss_curve.png')
        plt.close()

    def display_image(epoch_no):
      return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

    # plt.imshow(display_image(epochs))
    # plt.axis('off')# Display images

    anim_file = './results/cvae.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
      filenames = glob.glob('./results/image*.png')
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

  # import IPython
  # if IPython.version_info >= (6,2,0,''):
  #   display.Image(filename=anim_file)

  elif args.command == "evaluate":
    model.load_weights('./checkpoints/cvae_rectangle_checkpoint')


  else:
    print("'{}' is not recognized. Use 'train' or 'evaluate'".format(args.command))



if __name__ == '__main__':
  # parse arguments
  args = argparse_args()
  if args is None:
      exit()

  # main
  main(args)