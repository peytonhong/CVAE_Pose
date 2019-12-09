## CVAE_Pose
Convolutional Variational Autoencoder for object pose estimation (Pytorch version, Under construction)

**FC layer is added on latent vectors (z) to estimate rotation angle.**

* Input image
  * Rectangles with various rotation.
* Output image
  * Reconstructed rectangle images
* Output (FC)
  * Rotation angle (pose)

# Reconstructed image
![Reconstructed image](https://github.com/peytonhong/CVAE_Pose/blob/cae_pytorch/results/image_at_epoch_0099.png)

# 1D pose estimation result
![1D Pose estimation z_dim 2](https://github.com/peytonhong/CVAE_Pose/blob/cae_pytorch/results/pose_result_z_dim_2.png)

Pose loss of AE at z_dim=2 is about 0.1 deg.

Pose loss of VAE at z_dim=2 is about 0.07 deg.

This result shows that VAE is more accurate than AE for 1d pose estimation. (But both results are actually very similar.)
