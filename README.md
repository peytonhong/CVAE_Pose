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
![Reconstructed image_ae](https://github.com/peytonhong/CVAE_Pose/blob/master/results/image_at_epoch_0009.png)
* Description
* Left: Training results
* Right: Test results.
* 1st row: Input images
* 2nd row: Ground truth
* 3rd row: Reconstructed images
