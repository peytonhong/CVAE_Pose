## CVAE_Pose
Convolutional Variational Autoencoder for object pose estimation (Pytorch version, Under construction)

**FC layer is added on latent vectors (z) to estimate rotation angle.**

* Input image
  * Rectangles with various rotation.
* Output image
  * Reconstructed rectangle images
* Output (FC)
  * Rotation matrix (3x3)

# Reconstructed image
![Reconstructed image_vae](https://github.com/peytonhong/CVAE_Pose/blob/cvae_lm/results/image_at_epoch_0299.png)
VAE result

![Reconstructed image_ae](https://github.com/peytonhong/CVAE_Pose/blob/cvae_lm/results/image_at_epoch_0178_ae.png)
AE result

# 3X3 rotation matrix estimation result
![R matrix estimation z_dim 9 vae](https://github.com/peytonhong/CVAE_Pose/blob/cvae_lm/results/pose_result_vae_zdim9.png)
VAE result

![R matrix estimation z_dim 9 ae](https://github.com/peytonhong/CVAE_Pose/blob/cvae_lm/results/pose_result_ae_zdim9.png)
AE result

R matrix error of VAE at z_dim=9 is about 0.0176

R matrix error of AE at z_dim=9 is smaller than VAE version. (number not shown due to data lost)
