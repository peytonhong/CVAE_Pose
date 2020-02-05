# CVAE_Pose
Convolutional (Variational) Autoencoder for object pose estimation (Pytorch version, Under construction)

**FC layer is added on latent vectors (z) to estimate rotation matrix.**

**Pointcloud 3d model is rendered along with the estimated rotation angle. The rendered (and rotated) images are then evaluated with the output of decoder using MSE loss.**

### Network Architecture
![Architecture](https://github.com/peytonhong/CVAE_Pose/blob/master/docs/structure(rendering).png)

* Input image
  * LineMod dataset (duck) with various augmentation. (Random bubble, Random background, Random scale, Gamma correction)
* Output image
  * Reconstructed duck images
* Output (FC)
  * Rotation matrix (3x3)

### Reconstructed image (include rendering)
![Reconstructed image_ae](https://github.com/peytonhong/CVAE_Pose/blob/master/docs/image_at_epoch_9999(rendering).png)
### Reconstructed image (no rendering)
![Reconstructed image_ae (no rendering)](https://github.com/peytonhong/CVAE_Pose/blob/master/docs/image_at_epoch_9999(no_rendering).png)

##### Reconstructed Image Description
* Left: Training results
* Right: Test results.
* 1st row: Input images
* 2nd row: Ground truth
* 3rd row: Reconstructed images
* 4th row: Rendered (rotated) images

### Rotation loss comparison
* Without pointcloud rendering: 0.170
* With pointcloud rendering: 0.137

**This result shows that the pointcloud rendering is useful for pose estimation. However, the loss curve in rendering case is not stable since the renderer doesn't have gradients and the rendering result largely depends on initial condition of weight parameters.**


### Loss curve (include rendering)
![Loss curve_ae](https://github.com/peytonhong/CVAE_Pose/blob/master/docs/loss_curve(rendering).png)
### Loss curve (no rendering)
![Loss curve_ae (no rendering)](https://github.com/peytonhong/CVAE_Pose/blob/master/docs/loss_curve(no_rendering).png)
