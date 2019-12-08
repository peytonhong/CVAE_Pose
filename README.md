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
![1D Pose estimation z_dim 2](https://github.com/peytonhong/CVAE_Pose/blob/master/results/pose_result_z_dim_2.png)

Pose loss is about 0.07 deg at z_dim=2

![1D Pose estimation z_dim 20](https://github.com/peytonhong/CVAE_Pose/blob/master/results/pose_result_z_dim_20.png)

Pose loss is about 0.09 deg at z_dim=20

Pose loss at z_dim=128 is large (60 deg). These results give a lesson that z_dim is the small the better.
