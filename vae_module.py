import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

# U-Net styled Extended Autoencoder
class Extended_AE(nn.Module):
    ''' Extended Autoencoder that includes skip connections as U-Net '''
    def __init__(self, input_dim, z_dim, max_channel, PoseNet):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            z_dim: A integer indicating the latent dimension.
        '''
        super().__init__()
        self.max_channel = int(max_channel)
        self.input_dim = input_dim
        # self.fc_input_dim = int((input_dim[0]/16)*(input_dim[1]/16)*self.max_channel)
        self.fc_input_dim = self.max_channel # for AveragePooling case
        self.z_dim = z_dim
        self.poseNet = PoseNet
        
        # Encoder
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3,                        out_channels=int(self.max_channel/8), kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=int(self.max_channel/8),  out_channels=int(self.max_channel/4), kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=int(self.max_channel/4),  out_channels=int(self.max_channel/2), kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(in_channels=int(self.max_channel/2),  out_channels=int(self.max_channel),   kernel_size=5, stride=2, padding=2)
        self.fc_enc = nn.Linear(in_features=self.fc_input_dim*4, out_features=z_dim)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.upconv = nn.ConvTranspose2d(in_channels=self.max_channel, out_channels=self.max_channel, kernel_size=4, stride=4, padding=0, output_padding=0)
        
        # self.bn1 = nn.BatchNorm2d(num_features=int(self.max_channel/8))
        # self.bn2 = nn.BatchNorm2d(num_features=int(self.max_channel/4))
        # self.bn3 = nn.BatchNorm2d(num_features=int(self.max_channel/2))
        # self.bn4 = nn.BatchNorm2d(num_features=int(self.max_channel))

        # Decoder
        self.fc_dec = nn.Linear(in_features=self.z_dim, out_features=self.fc_input_dim*4)
        self.dconv4 = nn.ConvTranspose2d(in_channels=int(self.max_channel*2),      out_channels=int(self.max_channel/2), kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dconv3 = nn.ConvTranspose2d(in_channels=int(self.max_channel*2/2),    out_channels=int(self.max_channel/4), kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dconv2 = nn.ConvTranspose2d(in_channels=int(self.max_channel*2/4),    out_channels=int(self.max_channel/8), kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dconv1 = nn.ConvTranspose2d(in_channels=int(self.max_channel*2/8),    out_channels=3,                       kernel_size=5, stride=2, padding=2, output_padding=1)

        # self.bn_d4 = nn.BatchNorm2d(num_features=int(self.max_channel/2))
        # self.bn_d3 = nn.BatchNorm2d(num_features=int(self.max_channel/4))
        # self.bn_d2 = nn.BatchNorm2d(num_features=int(self.max_channel/8))
        # self.bn_d1 = nn.BatchNorm2d(num_features=3)
        
    def forward(self, x):        
        # Encoder
        # x is of shape [batch_size, input_dim]
        # hidden = F.relu(self.linear(x))
        x_1 = F.relu(input=self.conv1(x))
        x_2 = F.relu(input=self.conv2(x_1))
        x_3 = F.relu(input=self.conv3(x_2))
        x_4 = F.relu(input=self.conv4(x_3))
        x_4_pooled = self.avgpool(x_4)
        z = self.fc_enc(x_4_pooled.flatten(start_dim=1))
        # z_mu = z[:, :self.z_dim]
        # z_var = z[:, self.z_dim:]        
        pose_est = self.poseNet(z)

        # Decoder
        # z_reshaped = self.fc_dec(z).view(-1, self.max_channel, int(self.input_dim[0]/16), int(self.input_dim[1]/16)) # [N, 512, 8, 8]
        z_reshaped = self.fc_dec(z)
        z_reshaped = F.relu(self.upconv(z_reshaped.view(-1, self.max_channel, 2, 2))) # [N, 512, 2, 2] --> [N, 512, 8, 8]
        d_4 = F.relu(self.dconv4(torch.cat((z_reshaped, x_4), dim=1)))
        d_3 = F.relu(self.dconv3(torch.cat((d_4, x_3), dim=1)))
        d_2 = F.relu(self.dconv2(torch.cat((d_3, x_2), dim=1)))
        d_1 = torch.sigmoid(self.dconv1(torch.cat((d_2, x_1), dim=1))) # reconstructed image

        return d_1, z, pose_est


class Encoder(nn.Module):
    ''' This the encoder part of VAE

    '''
    def __init__(self, input_dim, z_dim, max_channel):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            z_dim: A integer indicating the latent dimension.
        '''
        super().__init__()
        self.max_channel = int(max_channel)
        # self.linear = nn.Linear(input_dim, hidden_dim)
        # self.mu = nn.Linear(hidden_dim, z_dim)
        # self.var = nn.Linear(hidden_dim, z_dim)
        self.input_dim = input_dim
        self.fc_input_dim = int((input_dim[0]/16)*(input_dim[1]/16)*self.max_channel)
        self.z_dim = z_dim
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3,                        out_channels=int(self.max_channel/8), kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=int(self.max_channel/8),  out_channels=int(self.max_channel/4), kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=int(self.max_channel/4),  out_channels=int(self.max_channel/2), kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=int(self.max_channel/2),  out_channels=int(self.max_channel),   kernel_size=3, stride=1, padding=1)
        # self.z_mu = nn.Linear(in_features=self.fc_input_dim, out_features=z_dim)
        # self.z_var= nn.Linear(in_features=self.fc_input_dim, out_features=z_dim)
        self.fc= nn.Linear(in_features=self.fc_input_dim, out_features=z_dim)
        
    def forward(self, x):        
        # x is of shape [batch_size, input_dim]
        # hidden = F.relu(self.linear(x))
        x = self.maxpool(F.relu(input=self.conv1(x)))
        x = self.maxpool(F.relu(input=self.conv2(x)))
        x = self.maxpool(F.relu(input=self.conv3(x)))
        x = self.maxpool(F.relu(input=self.conv4(x)))
        x = x.view(-1, int(self.max_channel*(self.input_dim[0]/16)*(self.input_dim[1]/16)))
        # hidden is of shape [batch_size, hidden_dim]
        z = self.fc(x)
        z_mu = z[:, :self.z_dim]
        z_var = z[:, self.z_dim:]
        # z_mu is of shape [batch_size, latent_dim]
        # z_var = self.var(x)
        # z_var is of shape [batch_size, latent_dim]

        return z, z_mu, z_var

class Decoder(nn.Module):
    ''' This the decoder part of VAE

    '''
    def __init__(self, z_dim, output_dim, max_channel):
        '''
        Args:
            z_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the output dimension (in case of MNIST it is 28 * 28)
        '''
        super().__init__()
        self.max_channel = int(max_channel)
        self.output_dim = output_dim        
        self.fc_output_dim = int((output_dim[0]/16)*(output_dim[1]/16)*self.max_channel)
        # self.linear = nn.Linear(z_dim, hidden_dim)
        # self.out = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Linear(in_features=z_dim, out_features=self.fc_output_dim)
        self.dconv1 = nn.ConvTranspose2d(in_channels=int(self.max_channel),      out_channels=int(self.max_channel/2), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dconv2 = nn.ConvTranspose2d(in_channels=int(self.max_channel/2),    out_channels=int(self.max_channel/4), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dconv3 = nn.ConvTranspose2d(in_channels=int(self.max_channel/4),    out_channels=int(self.max_channel/8), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dconv4 = nn.ConvTranspose2d(in_channels=int(self.max_channel/8),    out_channels=1,   kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        # x is of shape [batch_size, latent_dim]
        x = self.fc(x)
        x = x.view(-1, self.max_channel, int(self.output_dim[0]/16), int(self.output_dim[1]/16))
        x = F.relu(self.dconv1(x))
        x = F.relu(self.dconv2(x))
        x = F.relu(self.dconv3(x))
        x = torch.sigmoid(self.dconv4(x))
        # hidden = F.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]

        # predicted = torch.sigmoid(self.out(hidden))
        # predicted is of shape [batch_size, output_dim]
        return x

class Pose(nn.Module):
    def __init__(self, z_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_features=z_dim, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=9)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        x = self.fc3(x) # (N,9)

        # Rotation matrix refinement to make det(R)=1 (referenced from Geonho Cha's paper)
        U, S, V = torch.svd(x.view(-1,3,3))
        R_hat = torch.matmul(U, V.transpose(1,2)) # R_hat = U*V', det(R_hat)=1 since U and V are orthogonal matrices.

        return R_hat.view(-1,9) # (N,9)


class VAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.

    '''
    def __init__(self, enc, dec, pose):
        super().__init__()

        self.enc = enc
        self.dec = dec
        self.pose = pose

    def forward(self, x):
        # encode
        z_mu, z_var = self.enc(x)
        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # decode
        predicted = self.dec(x_sample)

        # pose
        pose_estimate = self.pose(z_mu)

        return predicted, z_mu, z_var, pose_estimate


class AE(nn.Module):
    ''' This is the AE, which takes a encoder and decoder.
    '''
    def __init__(self, enc, dec, pose):
        super().__init__()

        self.enc = enc
        self.dec = dec
        self.pose = pose

    def forward(self, x):
        z, z_mu, z_var = self.enc(x)
        predicted = self.dec(z)
        pose_estimate = self.pose(z)

        return predicted, z_mu, z_var, pose_estimate


def to_polar(theta, theta_sym):
    sym_ratio = 360/theta_sym
    return torch.cat((torch.cos(theta*sym_ratio), torch.sin(theta*sym_ratio)), axis=0)

def generate_and_save_images(args, model, epoch, reconstructed_image_train, input_image_train, gt_image_train, rendered_imgs_train, reconstructed_image_test, input_image_test, gt_image_test, rendered_imgs_test):
    if reconstructed_image_train.shape[1] == 3: # color image [N, 3, H, W]
        reconstructed_image_train = reconstructed_image_train.cpu().detach().numpy().transpose([0,2,3,1])
        reconstructed_image_test = reconstructed_image_test.cpu().detach().numpy().transpose([0,2,3,1])
        gt_image_train = gt_image_train.cpu().detach().numpy().transpose([0,2,3,1])
        gt_image_test = gt_image_test.cpu().detach().numpy().transpose([0,2,3,1])
        rendered_imgs_train = rendered_imgs_train.transpose([0,2,3,1])
        rendered_imgs_test = rendered_imgs_test.transpose([0,2,3,1])
    elif reconstructed_image_train.shape[1] == 1: # mask image [N, 1, H, W] --> [N, H, W, 3]
        reconstructed_image_train = np.stack((reconstructed_image_train.cpu().detach().numpy().squeeze(),)*3, axis=-1)
        reconstructed_image_test = np.stack((reconstructed_image_test.cpu().detach().numpy().squeeze(),)*3, axis=-1)
        gt_image_train = np.stack((gt_image_train.cpu().detach().numpy().squeeze(),)*3, axis=-1)
        gt_image_test = np.stack((gt_image_test.cpu().detach().numpy().squeeze(),)*3, axis=-1)
        rendered_imgs_train = np.stack((rendered_imgs_train.squeeze(),)*3, axis=-1)
        rendered_imgs_test = np.stack((rendered_imgs_test.squeeze(),)*3, axis=-1)

    input_image_train = input_image_train.cpu().detach().numpy().transpose([0,2,3,1])
    
    
    input_image_test = input_image_test.cpu().detach().numpy().transpose([0,2,3,1])
    
    

    horizontal_gap_large = np.ones((128,32,3))
    horizontal_gap_small = np.ones((128,8,3))
    reconstructed_image_train_concat = np.hstack((reconstructed_image_train[0], horizontal_gap_small, reconstructed_image_train[1], horizontal_gap_small, reconstructed_image_train[2], horizontal_gap_small, reconstructed_image_train[3]))
    reconstructed_image_test_concat = np.hstack((reconstructed_image_test[0], horizontal_gap_small, reconstructed_image_test[1], horizontal_gap_small, reconstructed_image_test[2], horizontal_gap_small, reconstructed_image_test[3]))
    reconstructed_image_concat = np.hstack((reconstructed_image_train_concat, horizontal_gap_large, reconstructed_image_test_concat))

    input_image_train_concat = np.hstack((input_image_train[0], horizontal_gap_small, input_image_train[1], horizontal_gap_small, input_image_train[2], horizontal_gap_small, input_image_train[3]))
    input_image_test_concat = np.hstack((input_image_test[0], horizontal_gap_small, input_image_test[1], horizontal_gap_small, input_image_test[2], horizontal_gap_small, input_image_test[3]))
    input_image_concat = np.hstack((input_image_train_concat, horizontal_gap_large, input_image_test_concat))

    gt_image_train_concat = np.hstack((gt_image_train[0], horizontal_gap_small, gt_image_train[1], horizontal_gap_small, gt_image_train[2], horizontal_gap_small, gt_image_train[3]))
    gt_image_test_concat = np.hstack((gt_image_test[0], horizontal_gap_small, gt_image_test[1], horizontal_gap_small, gt_image_test[2], horizontal_gap_small, gt_image_test[3]))
    gt_image_concat = np.hstack((gt_image_train_concat, horizontal_gap_large, gt_image_test_concat))

    rendered_imgs_train_concat = np.hstack((rendered_imgs_train[0], horizontal_gap_small, rendered_imgs_train[1], horizontal_gap_small, rendered_imgs_train[2], horizontal_gap_small, rendered_imgs_train[3]))
    rendered_imgs_test_concat = np.hstack((rendered_imgs_test[0], horizontal_gap_small, rendered_imgs_test[1], horizontal_gap_small, rendered_imgs_test[2], horizontal_gap_small, rendered_imgs_test[3]))
    rendered_imgs_concat = np.hstack((rendered_imgs_train_concat, horizontal_gap_large, rendered_imgs_test_concat))
        
    if args.rendering:
        total_image = np.vstack((input_image_concat, gt_image_concat, reconstructed_image_concat, rendered_imgs_concat))
    else:
        total_image = np.vstack((input_image_concat, gt_image_concat, reconstructed_image_concat))

    if not os.path.isdir("./results"):
        os.mkdir("./results")

    fig = plt.figure()
    plt.imshow(total_image)
    plt.axis('off')
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
    plt.savefig('./results/image_at_epoch_{:04d}.png'.format(epoch), bbox_inches='tight')
    plt.close(fig)