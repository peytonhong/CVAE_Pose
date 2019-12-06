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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

BATCH_SIZE = 100     # number of data points in each batch
N_EPOCHS = 100       # times to run the model on complete data
INPUT_DIM = (128, 128) # size of each input (width, height)
LATENT_DIM = 2     # latent vector dimension
lr = 1e-3           # learning rate

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

def train():
    # set the train mode
    model.train()
    # loss of the epoch
    train_loss = 0
    for _ in range(1000):        
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

def test():
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

    return test_loss

best_test_loss = float('inf')

for e in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train()
    test_loss = test()
    
    end_time = time.time()
    
    train_loss /= 1000*BATCH_SIZE
    test_loss /= BATCH_SIZE

    print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}, Time per an epoch: {(end_time - start_time):.2f}')
    
    # reconstruction from random latent variable
    generate_and_save_images(model, e, random_vector_for_generation)

    if best_test_loss > test_loss:
        best_test_loss = test_loss
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
