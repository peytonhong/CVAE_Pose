import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vae_module import *
from rectangles import *
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

  return parser.parse_args()

def train(model, TRAIN_STEPS, BATCH_SIZE, device, optimizer):    
    # set the train mode
    model.train()
    # loss of the epoch
    train_loss = 0
    for _ in range(TRAIN_STEPS):        
        x, y, theta = generate_dataset(BATCH_SIZE)
        x = x.transpose([0,3,1,2])
        y = y.transpose([0,3,1,2])
        x, y, theta = torch.tensor(x), torch.tensor(y), torch.tensor(theta)
        x, y, theta = x.to(device), y.to(device), theta.to(device)
        # reshape the data into [batch_size, 784]
        # x = x.view(-1, INPUT_DIM)
        
        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        x_sample, z_mu, z_var, pose_est = model(x)
        # reconstruction loss : the lower the better (negative log likelihood)
        recon_loss = F.binary_cross_entropy(x_sample, y, reduction='sum')
        # kl divergence loss : the lower the better
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
        # pose loss    
        pose_est = pose_est.view(-1,)
        pose_est_polar = to_polar(pose_est, theta_sym=90)    
        pose_gt_polar = to_polar(theta, theta_sym=90)
        pose_loss = F.mse_loss(pose_est_polar, pose_gt_polar, reduction='mean')

        # ELBO (Evidence lower bound): the higher the better
        ELBO = recon_loss + kl_loss
        loss = ELBO + pose_loss # apply gradient descent (loss to be lower)

        # backward pass
        loss.backward()
        train_loss += loss.item()
        
        # update the weights
        optimizer.step()

    return train_loss

def test(model, BATCH_SIZE, device, generate_plot=False):
    # set the evaluation mode
    model.eval()
    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        x, y, theta = generate_dataset(BATCH_SIZE)
        x = x.transpose([0,3,1,2])
        y = y.transpose([0,3,1,2])
        # for i, (x, y, theta) in enumerate(generate_dataset(BATCH_SIZE)):
        # for x, y, theta in zip(test_x, test_y, test_theta):
        # reshape the data
        # x = x.view(-1, 28 * 28)
        x, y, theta = torch.tensor(x), torch.tensor(y), torch.tensor(theta)
        x, y, theta = x.to(device), y.to(device), theta.to(device)
        # forward pass
        x_sample, z_mu, z_var, pose_est = model(x)
        # reconstruction loss
        recon_loss = F.binary_cross_entropy(x_sample, y, reduction='sum')
        
        # kl divergence loss
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
        
        # pose loss
        pose_est = pose_est.view(-1,)
        pose_est_polar = to_polar(pose_est, theta_sym=90)    
        pose_gt_polar = to_polar(theta, theta_sym=90)
        pose_loss = F.mse_loss(pose_est_polar, pose_gt_polar, reduction='mean')

        # total loss
        ELBO = recon_loss + kl_loss
        loss = ELBO + pose_loss
        test_loss += loss.item()        
    
    if generate_plot:
        # Pose estimation result
        pose_result = np.hstack((theta.cpu().reshape(-1,1), pose_est.cpu().numpy().reshape(-1,1))) # [ground truth, estimated]
        # pose_result = pose_result[pose_result[:,0].argsort()]
        plt.plot([0,90], [0,90] ,'g')
        plt.scatter(pose_result[:,0]*180/np.pi, pose_result[:,1]*180/np.pi % 90)   # remnant of symmetric angle
        plt.xlabel('Angle [deg]')
        plt.ylabel('Angle [deg]')
        plt.legend(['Ground truth', 'Estimated'])
        plt.title('Rotation Angle Estimation Result')
        plt.grid()
        plt.savefig('./results/pose_result.png')
        plt.close()

    return test_loss, pose_loss


def main(args):    

    BATCH_SIZE = 100     # number of data points in each batch
    TRAIN_STEPS = 100    # number of train steps in each epoch. (number of data in each epoch = BATCH_SIZE * TRAIN_STEPS)
    N_EPOCHS = args.num_epochs       # times to run the model on complete data
    INPUT_DIM = (128, 128) # size of each input (width, height)
    LATENT_DIM = args.latent_dim     # latent vector dimension
    lr = 1e-3           # learning rate

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    random_vector_for_generation = torch.randn(size=[16, LATENT_DIM]).to(device)

    # encoder
    encoder = Encoder(INPUT_DIM, LATENT_DIM)

    # decoder
    decoder = Decoder(LATENT_DIM, INPUT_DIM)

    # pose
    poseNet = Pose(LATENT_DIM)

    # vae
    model = VAE(encoder, decoder, poseNet).to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if args.command == 'train':

        # transforms = transforms.Compose([transforms.ToTensor()])
        # train_dataset = datasets.MNIST(
        #     './data',
        #     train=True,
        #     download=True,
        #     transform=transforms)

        # test_dataset = datasets.MNIST(
        #     './data',
        #     train=False,
        #     download=True,
        #     transform=transforms
        # )
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

            train_loss = train(model, TRAIN_STEPS, BATCH_SIZE, device, optimizer)
            test_loss, pose_loss = test(model, BATCH_SIZE, device, generate_plot=False)
            
            end_time = time.time()
            
            train_loss /= TRAIN_STEPS*BATCH_SIZE
            test_loss /= BATCH_SIZE

            print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}, Pose Loss: {pose_loss*180/np.pi:.5f} [deg], Time per an epoch: {(end_time - start_time):.2f}')
            
            # reconstruction from random latent variable
            generate_and_save_images(model, e, random_vector_for_generation)

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

        BATCH_SIZE_TEST = 1000
        test_loss, pose_loss = test(model, BATCH_SIZE_TEST, device, generate_plot=True)
        test_loss /= BATCH_SIZE_TEST
        print(f'Test Loss: {test_loss:.2f}, Pose Loss: {pose_loss*180/np.pi:.5f} [deg]')
        
    else:
        print("'{}' is not recognized. Use 'train' or 'evaluate'".format(args.command))


if __name__ == '__main__':
  # parse arguments
  args = argparse_args()
  if args is None:
      exit()

  # main
  main(args)