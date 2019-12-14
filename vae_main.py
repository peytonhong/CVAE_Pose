import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vae_module import *
# from rectangles import *
from linemod_load import LineModDataset, ToTensor
import os
import glob
import time
import argparse

"""parsing and configuration"""
def argparse_args():  
  desc = "Pytorch implementation of 'Convolutional Augmented Variational AutoEncoder (CAVAE)'"
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('command', help="'train' or 'evaluate'")
  parser.add_argument('--latent_dim', default=2, type=int, help="Dimension of latent vector")
  parser.add_argument('--num_epochs', default=200, type=int, help="The number of epochs to run")
  parser.add_argument('--max_channel', default=512, type=int, help="The maximum number of channels in Encoder/Decoder")
  parser.add_argument('--vae_mode', default=False, type=bool, help="True: Enable Variational Autoencoder, False: Autoencoder")
  parser.add_argument('--plot_recon', default=False, type=bool, help="True: creates reconstructed image on each epoch")

  return parser.parse_args()

def train(model, dataset, device, optimizer, vae_mode):
    # set the train mode
    model.train()
    # loss of the epoch
    train_loss = 0
    for _, sampled_batch in enumerate(dataset):        
        x = sampled_batch['image_cropped']
        y = sampled_batch['image_cropped']
        pose_gt = sampled_batch['pose'] # (N,9)
        x, y, pose_gt = x.to(device), y.to(device), pose_gt.to(device)
                
        # update the gradients to zero
        optimizer.zero_grad()
        # forward pass
        x_sample, z_mu, z_var, pose_est = model(x)
        # reconstruction loss : the lower the better (negative log likelihood)
        # recon_loss = F.binary_cross_entropy(x_sample, y, reduction='sum')
        recon_loss = F.mse_loss(x_sample, y, reduction='mean')
        if vae_mode:    
            # kl divergence loss : the lower the better
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)

        # pose loss        
        # pose_est_polar = to_polar(pose_est, theta_sym=360)    
        # pose_gt_polar = to_polar(pose_gt, theta_sym=360)
        pose_loss = F.mse_loss(pose_est, pose_gt, reduction='mean')

        if vae_mode:
            # ELBO (Evidence lower bound): the higher the better
            ELBO = recon_loss + kl_loss
            loss = ELBO + pose_loss # apply gradient descent (loss to be lower)
        else:
            loss = recon_loss + pose_loss

        # backward pass
        loss.backward()
        train_loss += loss.item()
        
        # update the weights
        optimizer.step()

    return train_loss

def test(model, dataset, device, vae_mode, test_iter):
    # set the evaluation mode
    model.eval()
    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, sampled_batch in enumerate(dataset):        
            x = sampled_batch['image_cropped']
            y = sampled_batch['image_cropped']
            pose_gt = sampled_batch['pose'] # (N,9)
            x, y, pose_gt = x.to(device), y.to(device), pose_gt.to(device)
            # forward pass
            x_sample, z_mu, z_var, pose_est = model(x)
            # reconstruction loss
            # recon_loss = F.binary_cross_entropy(x_sample, y, reduction='sum')
            recon_loss = F.mse_loss(x_sample, y, reduction='mean')

            if vae_mode:
                # kl divergence loss
                kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
            
            # pose loss
            # pose_est_polar = to_polar(pose_est, theta_sym=360)    
            # pose_gt_polar = to_polar(pose_gt, theta_sym=360)
            pose_loss = F.mse_loss(pose_est, pose_gt, reduction='mean')

            if vae_mode:
                # total loss
                ELBO = recon_loss + kl_loss
                loss = ELBO + pose_loss
            else:
                loss = recon_loss + pose_loss

            test_loss += loss.item()        
            if i == test_iter:
                break
    

    # if generate_plot:
    #     # Pose estimation result
    #     pose_result = np.hstack((pose_gt.cpu().reshape(-1,1), pose_est.cpu().numpy().reshape(-1,1))) # [ground truth, estimated]
    #     # pose_result = pose_result[pose_result[:,0].argsort()]
    #     plt.plot([0,180], [0,180] ,'g')
    #     plt.scatter(pose_result[:,0]*180/np.pi, pose_result[:,1]*180/np.pi % 360)   # remnant of symmetric angle
    #     plt.xlabel('Angle [deg]')
    #     plt.ylabel('Angle [deg]')
    #     plt.legend(['Ground truth', 'Estimated'])
    #     plt.title('Rotation Angle Estimation Result')
    #     plt.grid()
    #     plt.savefig('./results/pose_result.png')
    #     plt.close()

    return test_loss, pose_loss, pose_est, pose_gt, x_sample


def main(args):    

    BATCH_SIZE = 60     # number of data points in each batch
    
    N_EPOCHS = args.num_epochs       # times to run the model on complete data
    INPUT_DIM = (128, 128) # size of each input (width, height)
    LATENT_DIM = args.latent_dim     # latent vector dimension
    lr = 1e-3           # learning rate
    max_channel = args.max_channel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    transform = transforms.Compose([ToTensor()])
    lm_dataset = LineModDataset(root_dir='D:\ImageDataset\PoseDataset\lm_full', object_number=9, transform=transform) # for duck object
    train_iterator = DataLoader(dataset=lm_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_iterator = DataLoader(dataset=lm_dataset, batch_size=BATCH_SIZE, shuffle=True)
    sample_iterator = DataLoader(dataset=lm_dataset, batch_size=16, shuffle=True)

    random_vector_for_generation = torch.randn(size=[16, LATENT_DIM]).to(device)

    # encoder
    encoder = Encoder(INPUT_DIM, LATENT_DIM, max_channel=max_channel)

    # decoder
    decoder = Decoder(LATENT_DIM, INPUT_DIM, max_channel=max_channel)

    # pose
    poseNet = Pose(LATENT_DIM)

    if args.vae_mode:
        # Variational Autoencoder
        model = VAE(encoder, decoder, poseNet).to(device)
    else:
        # Autoencoder
        model = AE(encoder, decoder, poseNet).to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
        
        best_test_loss = float('inf')
        loss_list = []

        for e in range(N_EPOCHS):

            start_time = time.time()

            train_loss = train(model, train_iterator, device, optimizer, vae_mode=args.vae_mode)
            test_loss, pose_loss, _, _, _ = test(model, test_iterator, device, vae_mode=args.vae_mode, test_iter=4)
            
            end_time = time.time()
            
            train_loss /= len(lm_dataset)
            test_loss /= BATCH_SIZE*4

            print(f'Epoch {e}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, R matrix Loss: {pose_loss:.8f}, Time per an epoch: {(end_time - start_time):.2f}')
                        
            if args.plot_recon:                
                # reconstruction from random latent variable
                _, _, _, _, reconstructed_image = test(model, sample_iterator, device, vae_mode=args.vae_mode, test_iter=0)
                generate_and_save_images(model, e, reconstructed_image)

            # save loss curve
            loss_list.append([e, train_loss, test_loss])
            plt.plot(np.array(loss_list)[:,0], np.array(loss_list)[:,1], marker='.')
            plt.plot(np.array(loss_list)[:,0], np.array(loss_list)[:,2], marker='.')
            plt.legend(['Train loss', 'Test loss'])
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid()
            plt.savefig('./results/loss_curve.png')            
            plt.close()            

            if best_test_loss > test_loss:
                best_test_loss = test_loss
                torch.save(model, './checkpoints/model_best.pth.tar')
                patience_counter = 1
            else:
                patience_counter += 1

            # if patience_counter > 3:
            #     break

        # sample and generate a image
        # z = torch.randn(1, LATENT_DIM).to(device)

        # run only the decoder
        # reconstructed_img = model.dec(z)
        # img = reconstructed_img.view(INPUT_DIM).data.cpu()

        # print(z.shape)
        # print(img.shape)

        # plt.imshow(img, cmap='gray')
        # plt.savefig('sample_image.png')
        # plt.close()
    elif args.command == 'evaluate':
        print(f'This is evaluation mode.')
        model = torch.load('./checkpoints/model_best.pth.tar')

        
        test_loss, pose_loss, _, _ = test(model, test_iterator, device, vae_mode=args.vae_mode, test_iter=4)
        test_loss /= BATCH_SIZE*4
        print(f'Test Loss: {test_loss:.8f}, R matrix Loss: {pose_loss:.8f}')

        # compute one sample for checking rotation matrix
        test_loss, pose_loss, pose_est, pose_gt = test(model, sample_iterator, device, vae_mode=args.vae_mode, test_iter=0)
        # print(pose_est.cpu())
        # print(pose_gt.cpu())

        
    else:
        print("'{}' is not recognized. Use 'train' or 'evaluate'".format(args.command))


if __name__ == '__main__':
  # parse arguments
  args = argparse_args()
  if args is None:
      exit()

  # main
  main(args)