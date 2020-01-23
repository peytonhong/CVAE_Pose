# CVAE_Pose
Convolutional (Variational) Autoencoder for object pose estimation (Pytorch version, Under construction)

**FC layer is added on latent vectors (z) to estimate rotation matrix.**

* Input image
  * LineMod dataset (duck) with various augmentation. (Random bubble, Random background, Random scale, Gamma correction)
* Output image
  * Reconstructed duck images
* Output (FC)
  * Rotation matrix (3x3)

### Reconstructed image
![Reconstructed image_ae](https://github.com/peytonhong/CVAE_Pose/blob/master/docs/image_at_epoch_9999.png)
##### Reconstructed Image Description
* Left: Training results
* Right: Test results.
* 1st row: Input images
* 2nd row: Ground truth
* 3rd row: Reconstructed images

### Rotation loss comparison
* Without pointcloud: 0.000137
