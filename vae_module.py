import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        self.fc= nn.Linear(in_features=self.fc_input_dim, out_features=z_dim*2)

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

        return z_mu, z_var

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
        self.dconv4 = nn.ConvTranspose2d(in_channels=int(self.max_channel/8),    out_channels=3,   kernel_size=3, stride=2, padding=1, output_padding=1)

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
        self.fc3 = nn.Linear(in_features=100, out_features=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        
        return x


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
        z_mu, z_var = self.enc(x)
        predicted = self.dec(z_mu)
        pose_estimate = self.pose(z_mu)

        return predicted, z_mu, z_var, pose_estimate


def to_polar(theta, theta_sym):
    sym_ratio = 360/theta_sym
    return torch.cat((torch.cos(theta*sym_ratio), torch.sin(theta*sym_ratio)), axis=0)

def generate_and_save_images(model, epoch, test_input):
    predictions = model.dec(test_input).cpu().detach().numpy()
    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i].transpose([1,2,0]), cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('./results/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()
    plt.close(fig)