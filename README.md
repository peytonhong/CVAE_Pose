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
![Reconstructed image](https://github.com/peytonhong/CVAE_Pose/blob/cvae_lm/results/reconstructed_duck.png)

# 1D pose estimation result
![1D Pose estimation z_dim 2](https://github.com/peytonhong/CVAE_Pose/blob/cvae_lm/results/pose_result_vae_z2.png)

Pose loss of VAE at z_dim=2 is about 0.04 deg.
