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
import configparser
from pathlib import Path
from tqdm import tqdm
import cv2

"""parsing and configuration"""
def argparse_args():  
  desc = "Pytorch implementation of 'Convolutional Augmented Variational AutoEncoder (CAVAE)'"
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument('command', help="'train' or 'evaluate'")
  parser.add_argument('--latent_dim', default=128, type=int, help="Dimension of latent vector")
  parser.add_argument('--num_epochs', default=50, type=int, help="The number of epochs to run")
  parser.add_argument('--batch_size', default=200, type=int, help="The number of batchs for each epoch")
  parser.add_argument('--max_channel', default=512, type=int, help="The maximum number of channels in Encoder/Decoder")
  parser.add_argument('--vae_mode', default=False, type=bool, help="True: Enable Variational Autoencoder, False: Autoencoder")
  parser.add_argument('--plot_recon', default=True, type=bool, help="True: creates reconstructed image on each epoch")
  parser.add_argument('--bootstrap', default=False, type=bool, help="True: Bootstrapped L2 loss for each pixel")

  return parser.parse_args()

def get_rendering(obj_model,rot_pose,tra_pose, ren):
    '''Convert pointcloud into 2D rendered image using given transformation matrix. Output: batch rendered images'''
    
    rendered_imgs = []
    for i in range(len(rot_pose)):
        ren.clear()
        M=np.eye(4)
        M[:3,:3]=rot_pose[i].reshape(3,3)
        M[:3,3]=tra_pose
        ren.draw_model(obj_model, M)
        img_r, depth_rend = ren.finish()
        img_r = img_r[:,:,::-1]
        vu_valid = np.where(depth_rend>0)
        bbox_gt = np.array([np.min(vu_valid[0]),np.min(vu_valid[1]),np.max(vu_valid[0]),np.max(vu_valid[1])])
        img_r = img_r[bbox_gt[0]:bbox_gt[2], bbox_gt[1]:bbox_gt[3]]
        img_r = cv2.resize(img_r, (128,128), interpolation=cv2.INTER_LINEAR)
        rendered_imgs.append(img_r)
    rendered_imgs = np.array(rendered_imgs, dtype=np.float32).transpose([0,3,1,2]) # [N,128,128,3] -> [N,3,128,128]
    # return img_r, depth_rend, bbox_gt
    return rendered_imgs

def bootstrapped_l2_loss(x, y, bootstrap_factor=4):
    ''' The Bootstrapped L2 Loss which is only computed on the pixels with the most biggest errors. '''
    _, c, h ,w = x.shape
    worst_pixels = h*w*bootstrap_factor
    mseloss = nn.MSELoss(reduction='none')
    batch_loss = 0
    for i in range(len(x)): # compute MSE loss for each image
        loss = mseloss(x[i], y[i]).sum(dim=0) # x[i]: [c, h, w] --> sum along with dim=0: [h, w]
        worst_loss = loss.view(-1).sort()[0][-worst_pixels:]
        batch_loss += torch.mean(worst_loss)
    batch_loss /= len(x)
    return batch_loss

def train(model, dataset, device, optimizer, epoch, args):
    # set the train mode
    model.train()
    # loss of the epoch
    train_loss = 0
    for _, sampled_batch in enumerate(tqdm(dataset, desc=f"Training with batch size ({args.batch_size})")):
        x = sampled_batch['image_aug']
        y = sampled_batch['image_gt_cropped']
        pose_gt = sampled_batch['pose'] # (N,9)
        x, y, pose_gt = x.to(device), y.to(device), pose_gt.to(device)
        start_time = time.time()
        # update the gradients to zero
        optimizer.zero_grad()
        # forward pass
        x_sample, z_mu, z_var, pose_est = model(x)
        # reconstruction loss : the lower the better (negative log likelihood)
        if args.bootstrap:        
            # bootstrap_factor = int((x.shape[-2]*x.shape[-1]) * (0.84**epoch))
            # bootstrap_factor = bootstrap_factor if bootstrap_factor > 4 else 4
            recon_loss = bootstrapped_l2_loss(x_sample, y, bootstrap_factor=4) # Bootstrapped L2 loss for the 4 biggest pixels            
        else:
            recon_loss = F.mse_loss(x_sample, y, reduction='mean')
        
        # reconstruction loss for checking the effect of bootstrapped l2 loss
        with torch.no_grad():
            recon_loss_full_pixel = F.mse_loss(x_sample, y, reduction='mean')
        
        if args.vae_mode:    
            # kl divergence loss : the lower the better
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)

        # pose loss        
        # pose_est_polar = to_polar(pose_est, theta_sym=360)    
        # pose_gt_polar = to_polar(pose_gt, theta_sym=360)
        pose_loss = F.mse_loss(pose_est, pose_gt, reduction='mean')

        # pointcloud rendering output loss
        rendered_imgs = get_rendering(dataset.dataset.obj_model, pose_est.cpu().detach().numpy(), dataset.dataset.cam_T, dataset.dataset.ren)
        rendering_loss = F.mse_loss(torch.tensor(rendered_imgs).to(device), x_sample, reduction='mean')

        if args.vae_mode:
            # ELBO (Evidence lower bound): the higher the better
            ELBO = recon_loss + kl_loss
            loss = ELBO + pose_loss # apply gradient descent (loss to be lower)
        else:
            loss = 0.1*recon_loss + 0.8*pose_loss + 0.1*rendering_loss

        # backward pass
        loss.backward()
        train_loss += loss.item()
        
        # update the weights
        optimizer.step()

    return train_loss, recon_loss, pose_loss, rendering_loss, recon_loss_full_pixel

def test(model, dataset, device, args, test_iter):
    # set the evaluation mode
    model.eval()
    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, sampled_batch in enumerate(tqdm(dataset, desc=f" Testing with batch size ({args.batch_size})")):        
            x = sampled_batch['image_cropped']
            y = sampled_batch['image_gt_cropped']
            image_aug = sampled_batch['image_aug']
            pose_gt = sampled_batch['pose'] # (N,9)
            x, y, pose_gt = x.to(device), y.to(device), pose_gt.to(device)
            # forward pass
            x_sample, z_mu, z_var, pose_est = model(x)
            # reconstruction loss
            recon_loss = F.mse_loss(x_sample, y, reduction='mean')
            if args.vae_mode:
                # kl divergence loss
                kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
            
            # pose loss
            # pose_est_polar = to_polar(pose_est, theta_sym=360)    
            # pose_gt_polar = to_polar(pose_gt, theta_sym=360)
            pose_loss = F.mse_loss(pose_est, pose_gt, reduction='mean')

            # pointcloud rendering output loss
            rendered_imgs = get_rendering(dataset.dataset.obj_model, pose_est.cpu().detach().numpy(), dataset.dataset.cam_T, dataset.dataset.ren)
            rendering_loss = F.mse_loss(torch.tensor(rendered_imgs).to(device), x_sample, reduction='mean')

            if args.vae_mode:
                # total loss
                ELBO = recon_loss + kl_loss
                loss = ELBO + pose_loss
            else:
                loss = 0.1*recon_loss + 0.8*pose_loss + 0.1*rendering_loss

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

    return test_loss, recon_loss, pose_loss, rendering_loss, pose_est, pose_gt, x_sample, x, y, image_aug, rendered_imgs


def main(args):    

    config = configparser.ConfigParser()
    config.read('./cfg/config.cfg')
    lm_path = Path(config['Dataset']['lm'])
    coco_path = Path(config['Dataset']['coco'])

    BATCH_SIZE = args.batch_size     # number of data points in each batch
    
    N_EPOCHS = args.num_epochs       # times to run the model on complete data
    INPUT_DIM = (128, 128) # size of each input (width, height)
    LATENT_DIM = args.latent_dim     # latent vector dimension
    lr = 2e-4           # learning rate
    max_channel = args.max_channel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    transform = transforms.Compose([ToTensor()])
    lm_dataset_train = LineModDataset(root_dir=lm_path, background_dir=coco_path, task='train', object_number=9, transform=transform, augmentation=True, use_offline_data=True) # for duck object
    lm_dataset_test = LineModDataset(root_dir=lm_path, background_dir=coco_path, task='test', object_number=9, transform=transform, augmentation=False, use_offline_data=True) # for duck object
    train_iterator = DataLoader(dataset=lm_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    test_iterator = DataLoader(dataset=lm_dataset_test, batch_size=BATCH_SIZE, shuffle=True)
    sample_iterator_train = DataLoader(dataset=lm_dataset_train, batch_size=4, shuffle=True)
    sample_iterator_test = DataLoader(dataset=lm_dataset_test, batch_size=4, shuffle=True)

    print(f'Train images:   {len(lm_dataset_train)}')
    print(f'Test images:    {len(lm_dataset_test)}')

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

    # DataParallel for Multi GPU
    model = nn.DataParallel(model).to(device)

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
            train_loss, recon_loss_train, pose_loss_train, rendering_loss_train, recon_loss_train_full = train(model, train_iterator, device, optimizer, e, args)
            train_time = time.time() - start_time
            start_time = time.time()
            test_loss, recon_loss_test, pose_loss_test, rendering_loss_test, _, _, _, _, _, _, _ = test(model, test_iterator, device, args, test_iter=None)            
            test_time = time.time() - start_time
            
            recon_loss_train /= len(lm_dataset_train)
            recon_loss_test /= len(lm_dataset_test)
            pose_loss_train /= len(lm_dataset_train)
            pose_loss_test /= len(lm_dataset_test)

            summary_note = f'Epoch: {e:3d}, Train Recon Loss: {recon_loss_train:.6f}, Test Recon Loss: {recon_loss_test:.6f}, R Loss train: {pose_loss_train:.6f}, R Loss test: {pose_loss_test:.6f}, Rendering Loss train: {rendering_loss_train:.6f}, Rendering Loss test: {rendering_loss_test:.6f}, Train Time: {(train_time):.2f}, Test Time: {(test_time):.2f}'
            print(summary_note)
            if e == 0:
                summary_file = open("results/summary_note.txt", 'w')
                summary_file.write(summary_note + '\n')
                summary_file.close()
            else:
                summary_file = open("results/summary_note.txt", 'a')
                summary_file.write(summary_note + '\n')
                summary_file.close()

            if args.plot_recon:
                # reconstruction from random latent variable
                _, _, _, _, _, _, reconstructed_image_train, input_image_train, gt_image_train, image_aug_train, rendered_imgs_train = test(model, sample_iterator_train, device, args, test_iter=0)
                _, _, _, _, _, _, reconstructed_image_test, input_image_test, gt_image_test, image_aug_test, rendered_imgs_test = test(model, sample_iterator_test, device, args, test_iter=0)
                generate_and_save_images(model, e, reconstructed_image_train, image_aug_train, gt_image_train, rendered_imgs_train, reconstructed_image_test, input_image_test, gt_image_test, rendered_imgs_test)

            # save loss curve
            loss_list.append([e, recon_loss_train, recon_loss_test, pose_loss_train, pose_loss_test])
            plt.plot(np.array(loss_list)[:,0], np.array(loss_list)[:,1], color='b', marker='.')
            plt.plot(np.array(loss_list)[:,0], np.array(loss_list)[:,2], color='g', marker='.')
            plt.plot(np.array(loss_list)[:,0], np.array(loss_list)[:,3], color='b', linestyle='--')
            plt.plot(np.array(loss_list)[:,0], np.array(loss_list)[:,4], color='g', linestyle='--')
            plt.legend(['Train recon loss', 'Test recon loss', 'Train R loss', 'Test R loss'])
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid()
            plt.savefig('./results/loss_curve.png')            
            plt.close()            

            if best_test_loss > pose_loss_test:
                best_test_loss = pose_loss_test
                torch.save(model, './checkpoints/model_best.pth.tar')
                patience_counter = 1
            else:
                patience_counter += 1

            # if patience_counter > 3:
            #     break

    elif args.command == 'evaluate':
        print(f'This is evaluation mode.')
        print(f'Total number of test images: {len(lm_dataset_test)}')
        model = torch.load('./checkpoints/model_best.pth.tar')

        # start_time = time.time()
        # test_loss, pose_loss, _, _, _, _, _ = test(model, test_iterator, device, vae_mode=args.vae_mode, test_iter=None)
        # test_time = time.time() - start_time
        # test_loss /= len(lm_dataset_test)
        # print(f'Test Loss: {test_loss:.6f}, R matrix Loss: {pose_loss:.6f}, Test Time: {(test_time):.2f}')

        # compute one sample for checking rotation matrix
        _, _, _, pose_est_train, pose_gt_train, reconstructed_image_train, input_image_train, gt_image_train, image_aug_train = test(model, sample_iterator_train, device, args, test_iter=0)
        _, _, _, pose_est_test, pose_gt_test, reconstructed_image_test, input_image_test, gt_image_test, image_aug_test = test(model, sample_iterator_test, device, args, test_iter=0)
        generate_and_save_images(model, 9999, reconstructed_image_train, image_aug_train, gt_image_train, reconstructed_image_test, input_image_test, gt_image_test)
        
        pose_train = np.hstack((pose_gt_train[0].cpu().numpy().transpose().reshape((-1,1)), pose_est_train[0].cpu().numpy().transpose().reshape((-1,1))))
        pose_test = np.hstack((pose_gt_test[0].cpu().numpy().transpose().reshape((-1,1)), pose_est_test[0].cpu().numpy().transpose().reshape((-1,1))))
        print(f'Pose estimation result (train: [GT , Estimation])')
        print(f'{pose_train}')
        print(f'Pose estimation result (test: [GT , Estimation])')
        print(f'{pose_test}')
        
    else:
        print("'{}' is not recognized. Use 'train' or 'evaluate'".format(args.command))


if __name__ == '__main__':
  # parse arguments
  args = argparse_args()
  if args is None:
      exit()

  # main
  main(args)