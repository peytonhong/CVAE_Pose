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
![Reconstructed image](https://github.com/peytonhong/CVAE_Pose/blob/cvae_pytorch/results/image_at_epoch_0199.png)

# 1D pose estimation result
![1D Pose estimation](https://github.com/peytonhong/CVAE_Pose/blob/cvae_pytorch/results/pose_result_z_dim_2.png)

